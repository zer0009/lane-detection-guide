import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'dart:typed_data';
import 'dart:math' show atan2, cos, max, min, pi, sqrt;
import 'package:flutter/foundation.dart' show compute, debugPrint;
import 'dart:async';

class CameraParameters {
  final double focalLength;
  final cv.Point principalPoint;
  final double height;

  CameraParameters({
    required this.focalLength,
    required this.principalPoint,
    required this.height,
  });
}

class LaneDetectionResult {
  final double deviation;
  final Uint8List processedImage;
  final bool isLaneDetected;
  final double lateralOffset; // Meters from lane center (negative = left)
  final double orientation; // Degrees from straight ahead

  LaneDetectionResult(
    this.deviation, 
    this.processedImage, 
    this.isLaneDetected, {
    this.lateralOffset = 0.0,
    this.orientation = 0.0,
  });
}

class LaneDetector {
  static cv.Mat? _cachedMask;
  static cv.Size? _lastSize;
  
  // Adjust these constants for better performance/accuracy balance
  static const double PROCESSING_SCALE = 0.2; // Reduced scale for faster processing
  static const int SKIP_FRAMES = 2; // Process more frames
  static int _frameCounter = 0;
  static LaneDetectionResult? _lastResult;
  static DateTime? _lastProcessTime;
  static const Duration FORCE_PROCESS_INTERVAL = Duration(milliseconds: 250); // Faster updates

  static const int HISTORY_SIZE = 3; // Reduced history size for faster response
  static List<double> _deviationHistory = [];
  static List<bool> _detectionHistory = [];
  static double _lastStableDeviation = 0.0;
  static const double CONFIDENCE_THRESHOLD = 0.6;

  // Add new constants for improved line detection
  static const double MIN_LINE_LENGTH_RATIO = 0.12; // Minimum line length as ratio of height
  static const double MAX_LINE_LENGTH_RATIO = 0.7;  // Maximum line length as ratio of height
  static const double MIN_CONFIDENCE_POINTS = 3;    // Minimum points needed for confidence

  // Add new constants for lane tracking
  static const double LANE_WIDTH_METERS = 3.7; // Standard lane width in meters
  static const double TYPICAL_CAMERA_HEIGHT = 1.5; // Typical phone height in meters
  static double? _lastValidLaneWidth; // Store last known good lane width in pixels
  static cv.Point? _lastValidVanishingPoint; // Track vanishing point for orientation
  
  // Add camera calibration parameters
  static final _cameraParams = CameraParameters(
    focalLength: 1000.0, // Approximate focal length in pixels
    principalPoint: cv.Point(0, 0), // Will be set based on image size
    height: TYPICAL_CAMERA_HEIGHT,
  );

  static cv.Mat _createROIMask(int width, int height) {
    if (_cachedMask != null && 
        _lastSize != null && 
        _lastSize!.width == width && 
        _lastSize!.height == height) {
      return _cachedMask!;
    }

    final mask = cv.Mat.zeros(height, width, cv.MatType.CV_8UC1);
    final roiPoints = [
      cv.Point(0, height),                           
      cv.Point(width, height),                       
      cv.Point((width * 0.85).toInt(), (height * 0.5).toInt()),  // Adjusted ROI
      cv.Point((width * 0.15).toInt(), (height * 0.5).toInt()),  // Adjusted ROI
    ];
    
    final roiContours = cv.VecVecPoint.fromList([roiPoints]);
    cv.fillPoly(mask, roiContours, cv.Scalar.all(255));
    
    _cachedMask?.dispose();
    _cachedMask = mask;
    _lastSize = cv.Size(width, height);
    
    return mask;
  }

  static Future<LaneDetectionResult> processFrame(Uint8List imageBytes) async {
    // Optimize frame processing logic
    _frameCounter = (_frameCounter + 1) % (SKIP_FRAMES + 1);
    final now = DateTime.now();
    final shouldForceProcess = _lastProcessTime == null || 
        now.difference(_lastProcessTime!) > FORCE_PROCESS_INTERVAL;
    
    if (_frameCounter != 0 && !shouldForceProcess && _lastResult != null) {
      return _lastResult!;
    }
    _lastProcessTime = now;

    final List<cv.Mat> resources = [];
    
    try {
      // Optimize image processing pipeline
      final img = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      resources.add(img);
      
      final height = img.rows;
      final width = img.cols;
      final centerX = width / 2;

      // Smaller processing size
      final processWidth = (width * PROCESSING_SCALE).toInt();
      final processHeight = (height * PROCESSING_SCALE).toInt();
      final resized = cv.resize(img, (processWidth, processHeight));
      resources.add(resized);

      // Create visualization output with lower quality for better performance
      final visualOutput = cv.resize(img, (width ~/ 1.5, height ~/ 1.5)); // Reduced output size
      resources.add(visualOutput);

      // Optimize color conversion and edge detection
      final gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY);
      resources.add(gray);
      
      // Simplified edge detection
      final blurred = cv.gaussianBlur(gray, (3, 3), 1.0); // Smaller kernel
      final edges = cv.canny(blurred, 50, 150, apertureSize: 3);
      resources.add(edges);

      // Simplified color detection
      final hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV);
      resources.add(hsv);

      // Only use white and yellow masks for better performance
      final whiteMask = _createColorMask(hsv, 'white');
      final yellowMask = _createColorMask(hsv, 'yellow');
      resources.addAll([whiteMask, yellowMask]);

      // Combine masks more efficiently
      final combinedMask = cv.bitwiseOR(whiteMask, yellowMask);
      resources.add(combinedMask);

      // Simplified edge combination
      cv.bitwiseOR(edges, combinedMask, dst: edges);

      // Apply ROI mask
      final mask = _createROIMask(processWidth, processHeight);
      cv.bitwiseAND(edges, mask, dst: edges);

      // Optimize Hough transform parameters
      final lines = cv.HoughLinesP(
        edges,
        1.0,         
        pi / 180.0,  
        20,          // Reduced threshold
        minLineLength: 20.0,  // Shorter minimum length
        maxLineGap: 30.0      // Increased gap for better connection
      );

      // Enhanced line filtering with stricter criteria
      final List<List<cv.Point>> leftSegments = [];
      final List<List<cv.Point>> rightSegments = [];
      const double CLUSTER_DISTANCE = 20.0; // Reduced clustering distance
      
      for (int i = 0; i < lines.rows; i++) {
        final line = lines.row(i);
        final x1 = line.at<int>(0, 0).toDouble() / PROCESSING_SCALE;
        final y1 = line.at<int>(0, 1).toDouble() / PROCESSING_SCALE;
        final x2 = line.at<int>(0, 2).toDouble() / PROCESSING_SCALE;
        final y2 = line.at<int>(0, 3).toDouble() / PROCESSING_SCALE;
        
        final points = [cv.Point(x1.toInt(), y1.toInt()), cv.Point(x2.toInt(), y2.toInt())];
        final avgX = (x1 + x2) / 2;
        final avgY = (y1 + y2) / 2;
        
        // Stricter angle and position filtering
        final dy = (y2 - y1);
        final dx = (x2 - x1);
        final angle = dy != 0 ? (dx / dy).abs() : double.infinity;
        final length = sqrt(dx * dx + dy * dy);
        
        // More restrictive conditions for valid lane lines
        if (angle < 2.0 && // More permissive angle threshold
            length > height * 0.12 && // Reduced minimum length requirement
            avgY > height * 0.5 && // Look higher up in the image
            avgY < height * 0.95) { // Ignore lines too close to bottom
          
          if (avgX < centerX && avgX > width * 0.02) { // More permissive left boundary
            _addToSegmentCluster(leftSegments, points, CLUSTER_DISTANCE * 1.5); // Increased clustering distance
          } else if (avgX > centerX && avgX < width * 0.98) { // More permissive right boundary
            _addToSegmentCluster(rightSegments, points, CLUSTER_DISTANCE * 1.5);
          }

          // Visualize detected segments
          cv.line(
            visualOutput,
            cv.Point(x1.toInt(), y1.toInt()),
            cv.Point(x2.toInt(), y2.toInt()),
            avgX < centerX ? cv.Scalar(255, 0, 0) : cv.Scalar(0, 0, 255),
            thickness: 2
          );
        }
      }

      // Filter out small clusters and calculate lane center
      leftSegments.removeWhere((cluster) => cluster.length < 2);
      rightSegments.removeWhere((cluster) => cluster.length < 2);

      double laneCenter;
      bool isValidLane = false;

      if (leftSegments.isNotEmpty && rightSegments.isNotEmpty) {
        // Calculate weighted average based on segment point count
        double leftX = 0;
        double leftWeight = 0;
        for (var segment in leftSegments.first) {
          final weight = 1.0;  // Equal weight for each point
          leftX += segment.x * weight;
          leftWeight += weight;
        }
        leftX /= leftWeight;

        double rightX = 0;
        double rightWeight = 0;
        for (var segment in rightSegments.first) {
          final weight = 1.0;
          rightX += segment.x * weight;
          rightWeight += weight;
        }
        rightX /= rightWeight;

        // Stricter validation criteria for lane detection
        final laneWidth = rightX - leftX;
        if (laneWidth > width * 0.3 && laneWidth < width * 0.8) {  // Stricter lane width range
          laneCenter = (leftX + rightX) / 2;
          isValidLane = leftSegments.first.length >= 4 && rightSegments.first.length >= 4; // Require more points
        } else {
          laneCenter = centerX;
          isValidLane = false;
        }
      } else if (leftSegments.isNotEmpty) {
        double avgX = 0;
        double avgY = 0;
        for (var point in leftSegments.first) {
          avgX += point.x;
          avgY += point.y;
        }
        final leftX = avgX / leftSegments.first.length;
        final leftY = avgY / leftSegments.first.length;
        
        // Estimate lane center based on line angle and position
        final dy = height - leftY;
        final dx = leftSegments.first.last.x - leftSegments.first.first.x;
        final angle = atan2(dy, dx);
        laneCenter = leftX + (width * 0.35 * cos(angle)); // Dynamic offset based on line angle
        isValidLane = leftSegments.first.length >= 2; // Only need 2 points for validation
      } else if (rightSegments.isNotEmpty) {
        double avgX = 0;
        double avgY = 0;
        for (var point in rightSegments.first) {
          avgX += point.x;
          avgY += point.y;
        }
        final rightX = avgX / rightSegments.first.length;
        final rightY = avgY / rightSegments.first.length;
        
        // Similar angle-based estimation for right lane
        final dy = height - rightY;
        final dx = rightSegments.first.last.x - rightSegments.first.first.x;
        final angle = atan2(dy, dx);
        laneCenter = rightX - (width * 0.35 * cos(angle));
        isValidLane = rightSegments.first.length >= 2;
      } else {
        laneCenter = centerX;
      }

      // Draw guidance lines
      cv.line(
        visualOutput,
        cv.Point(centerX.toInt(), height),
        cv.Point(centerX.toInt(), (height * 0.6).toInt()),
        cv.Scalar(255, 255, 255),
        thickness: 2
      );

      cv.line(
        visualOutput,
        cv.Point(laneCenter.toInt(), height),
        cv.Point(laneCenter.toInt(), (height * 0.6).toInt()),
        cv.Scalar(0, 255, 0),
        thickness: 2
      );

      // Calculate deviation only if lane detection is valid
      double deviation = (laneCenter - centerX) / (width / 2);
      deviation = deviation.clamp(-1.0, 1.0);

      // Enhanced confidence calculation
      double calculateConfidence(List<List<cv.Point>> segments) {
        if (segments.isEmpty) return 0.0;
        
        final points = segments.first.length;
        final avgLength = _calculateAverageSegmentLength(segments.first);
        
        // More lenient confidence calculation
        final minPoints = 2; // Reduced minimum points
        final lengthFactor = avgLength / (height * MIN_LINE_LENGTH_RATIO);
        return min(1.0, (points / minPoints) * lengthFactor * 1.5); // Increased multiplier
      }

      // Calculate confidence for both sides
      final leftConfidence = calculateConfidence(leftSegments);
      final rightConfidence = calculateConfidence(rightSegments);
      
      // Use weighted confidence for final detection
      final confidence = (leftConfidence + rightConfidence) / 2;
      isValidLane = confidence > CONFIDENCE_THRESHOLD;

      // Apply temporal smoothing
      _deviationHistory.add(deviation);
      _detectionHistory.add(isValidLane && confidence > CONFIDENCE_THRESHOLD);
      
      if (_deviationHistory.length > HISTORY_SIZE) {
        _deviationHistory.removeAt(0);
        _detectionHistory.removeAt(0);
      }

      // Calculate smoothed values
      if (_deviationHistory.isNotEmpty) {
        double sum = 0;
        int validCount = 0;
        
        for (int i = 0; i < _deviationHistory.length; i++) {
          if (_detectionHistory[i]) {
            sum += _deviationHistory[i];
            validCount++;
          }
        }

        if (validCount > 0) {
          deviation = sum / validCount;
          _lastStableDeviation = deviation;
        } else {
          deviation = _lastStableDeviation;
          isValidLane = false;
        }
      }

      _lastResult = LaneDetectionResult(
        deviation,
        cv.imencode('.jpg', visualOutput).$2,
        isValidLane && confidence > CONFIDENCE_THRESHOLD
      );
      return _lastResult!;

    } catch (e) {
      debugPrint('Error processing frame: $e');
      _lastResult = LaneDetectionResult(0.0, imageBytes, false);
      return _lastResult!;
    } finally {
      // Clean up resources
      for (final mat in resources) {
        mat.dispose();
      }
    }
  }

  // Add new helper method for segment clustering
  static void _addToSegmentCluster(List<List<cv.Point>> clusters, List<cv.Point> points, double threshold) {
    bool addedToCluster = false;
    
    for (var cluster in clusters) {
      // Check if any point in the new segment is close to any point in the cluster
      for (var newPoint in points) {
        for (var clusterPoint in cluster) {
          final dx = newPoint.x - clusterPoint.x;
          final dy = newPoint.y - clusterPoint.y;
          final distance = sqrt(dx * dx + dy * dy);
          
          if (distance < threshold) {
            cluster.addAll(points);
            addedToCluster = true;
            break;
          }
        }
        if (addedToCluster) break;
      }
      if (addedToCluster) break;
    }
    
    if (!addedToCluster) {
      clusters.add(points);
    }
  }

  static void dispose() {
    _cachedMask?.dispose();
    _cachedMask = null;
    _lastSize = null;
    _lastResult = null;
  }

  // Add memory management
  static void clearCache() {
    _lastResult = null;
    _lastProcessTime = null;
    _frameCounter = 0;
    _deviationHistory.clear();
    _detectionHistory.clear();
    _lastStableDeviation = 0.0;
  }

  // Enhanced adaptive thresholding based on lighting conditions
  static double _calculateThreshold(cv.Mat gray) {
    final mean = cv.mean(gray);
    final stdDev = cv.meanStdDev(gray);
    
    // Get the mean brightness (first channel)
    final meanBrightness = mean.val[0];
    
    // Get the standard deviation value
    final standardDeviation = stdDev.$2.val[0];
    
    // Base threshold calculation using mean brightness
    double baseThreshold = meanBrightness < 127 ? 30.0 : 50.0;
    
    // Adjust threshold based on image contrast (standard deviation)
    if (standardDeviation < 30) {
      // Low contrast image - reduce threshold
      baseThreshold *= 0.8;
    } else if (standardDeviation > 60) {
      // High contrast image - increase threshold
      baseThreshold *= 1.2;
    }
    
    // Ensure threshold stays within reasonable bounds
    return baseThreshold.clamp(20.0, 70.0);
  }

  // Improve line filtering
  static bool _isValidLine(double x1, double y1, double x2, double y2, int width, int height) {
    final dy = (y2 - y1);
    final dx = (x2 - x1);
    final angle = dy != 0 ? (dx / dy).abs() : double.infinity;
    final length = sqrt(dx * dx + dy * dy);
    
    return angle < 2.5 && // More permissive angle
           length > height * 0.1 && // Minimum length
           length < height * 0.8;   // Maximum length
  }

  // Optimize color mask creation
  static cv.Mat _createColorMask(cv.Mat hsv, String type) {
    final mask = cv.Mat.zeros(hsv.rows, hsv.cols, cv.MatType.CV_8UC1);
    
    try {
      switch (type) {
        case 'white':
          // Simplified white detection
          cv.inRange(
            hsv, 
            cv.Mat.fromList(1, 3, cv.MatType.CV_8UC1, [0, 0, 215]), 
            cv.Mat.fromList(1, 3, cv.MatType.CV_8UC1, [180, 30, 255]), 
            dst: mask
          );
          break;

        case 'yellow':
          // Simplified yellow detection
          cv.inRange(
            hsv, 
            cv.Mat.fromList(1, 3, cv.MatType.CV_8UC1, [15, 120, 120]), 
            cv.Mat.fromList(1, 3, cv.MatType.CV_8UC1, [40, 255, 255]), 
            dst: mask
          );
          break;

        default:
          throw Exception('Invalid color mask type');
      }

      // Simplified noise reduction
      final kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3));
      cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, dst: mask);
      kernel.dispose();

      return mask;
    } catch (e) {
      mask.dispose();
      rethrow;
    }
  }

  // New helper method for segment length calculation
  static double _calculateAverageSegmentLength(List<cv.Point> points) {
    if (points.length < 2) return 0.0;
    
    double totalLength = 0.0;
    for (int i = 0; i < points.length - 1; i++) {
      final dx = points[i + 1].x - points[i].x;
      final dy = points[i + 1].y - points[i].y;
      totalLength += sqrt(dx * dx + dy * dy);
    }
    
    return totalLength / (points.length - 1);
  }

  static cv.Point? _findIntersection(cv.Point p1, cv.Point p2, cv.Point p3, cv.Point p4) {
    // Line 1 represented as a1x + b1y = c1
    final a1 = p2.y - p1.y;
    final b1 = p1.x - p2.x;
    final c1 = a1 * p1.x + b1 * p1.y;

    // Line 2 represented as a2x + b2y = c2
    final a2 = p4.y - p3.y;
    final b2 = p3.x - p4.x;
    final c2 = a2 * p3.x + b2 * p3.y;

    final determinant = a1 * b2 - a2 * b1;

    if (determinant == 0) {
      // Lines are parallel
      return null;
    }

    final x = (b2 * c1 - b1 * c2) / determinant;
    final y = (a1 * c2 - a2 * c1) / determinant;

    return cv.Point(x.toInt(), y.toInt());
  }
}