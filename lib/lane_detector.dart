import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'dart:typed_data';
import 'dart:math' show pi, min, max, sqrt;
import 'package:flutter/foundation.dart' show debugPrint;

class LaneDetectionResult {
  final double deviation;
  final Uint8List processedImage;
  final bool isLaneDetected;

  LaneDetectionResult(this.deviation, this.processedImage, this.isLaneDetected);
}

class LaneDetector {
  static cv.Mat? _cachedMask;
  static cv.Size? _lastSize;
  
  // Increase frame skipping and reduce processing size for better performance
  static const double PROCESSING_SCALE = 0.25; // Reduce to 25% of original size
  static const int SKIP_FRAMES = 3; // Process every 4th frame
  static int _frameCounter = 0;
  static LaneDetectionResult? _lastResult;
  static DateTime? _lastProcessTime;
  static const Duration FORCE_PROCESS_INTERVAL = Duration(milliseconds: 500);

  static const int HISTORY_SIZE = 5;
  static List<double> _deviationHistory = [];
  static List<bool> _detectionHistory = [];
  static double _lastStableDeviation = 0.0;
  static const double CONFIDENCE_THRESHOLD = 0.6;

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
    // Skip frames but ensure we process at least every 500ms
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
      // Convert image bytes to Mat
      final img = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
      resources.add(img);
      
      final height = img.rows;
      final width = img.cols;
      
      // Define center X coordinate
      final centerX = width / 2;

      // Resize image for faster processing
      final processWidth = (width * PROCESSING_SCALE).toInt();
      final processHeight = (height * PROCESSING_SCALE).toInt();
      final resized = cv.resize(img, (processWidth, processHeight));
      resources.add(resized);

      // Create visualization output (only if needed)
      final visualOutput = img.clone();
      resources.add(visualOutput);

      // Convert to grayscale
      final gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY);
      resources.add(gray);
      
      // Apply blur and threshold in one step for better performance
      final blurred = cv.gaussianBlur(gray, (5, 5), 1.5);
      final adaptiveThresh = cv.adaptiveThreshold(
        blurred,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        11,  // Block size
        2    // C constant
      );
      resources.add(adaptiveThresh);

      // Use adaptive threshold result for edge detection
      final edges = cv.canny(blurred, 50, 150);
      resources.add(edges);

      // After creating resized image, add color-based detection
      final hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV);
      resources.add(hsv);

      // Create masks for different common line colors (white, yellow)
      final whiteMask = cv.inRange(
        hsv,
        cv.Mat.zeros(1, 1, cv.MatType.CV_8UC3)..setTo(cv.Scalar(0, 0, 180)),    // Low white in HSV
        cv.Mat.zeros(1, 1, cv.MatType.CV_8UC3)..setTo(cv.Scalar(180, 30, 255))  // High white in HSV
      );

      final yellowMask = cv.inRange(
        hsv,
        cv.Mat.zeros(1, 1, cv.MatType.CV_8UC3)..setTo(cv.Scalar(15, 80, 120)),   // Low yellow in HSV
        cv.Mat.zeros(1, 1, cv.MatType.CV_8UC3)..setTo(cv.Scalar(35, 255, 255))   // High yellow in HSV
      );

      // Combine masks using bitwise operations
      final combinedMask = cv.bitwiseOR(whiteMask, yellowMask);
      resources.add(combinedMask);

      // Enhanced edge detection with color information
      final combinedEdges = cv.Mat.zeros(processHeight, processWidth, cv.MatType.CV_8UC1);
      resources.add(combinedEdges);
      
      // Combine traditional edge detection with color information
      final cannyEdges = cv.canny(cv.gaussianBlur(cv.cvtColor(resized, cv.COLOR_BGR2GRAY), (5, 5), 1.5), 30, 90);
      cv.bitwiseOR(cannyEdges, combinedMask, dst: combinedEdges);

      // Apply ROI mask
      final mask = _createROIMask(processWidth, processHeight);
      final maskedEdges = cv.Mat.zeros(processHeight, processWidth, cv.MatType.CV_8UC1);
      resources.add(maskedEdges);
      cv.bitwiseAND(combinedEdges, mask, dst: maskedEdges);

      // Use probabilistic Hough transform with optimized parameters for curved lines
      final lines = cv.HoughLinesP(
        maskedEdges,
        1.0,         // Distance resolution (pixels)
        pi / 180.0,  // Angle resolution (radians)
        15,          // Lower threshold for better detection of broken lines
        minLineLength: 20.0,  // Shorter minimum length to catch curve segments
        maxLineGap: 30.0      // Larger gap to connect broken segments
      );

      // Enhanced line filtering and clustering for curves
      final List<List<cv.Point>> leftSegments = [];
      final List<List<cv.Point>> rightSegments = [];
      const double CLUSTER_DISTANCE = 30.0; // pixels
      
      for (int i = 0; i < lines.rows; i++) {
        final line = lines.row(i);
        final x1 = line.at<int>(0, 0).toDouble() / PROCESSING_SCALE;
        final y1 = line.at<int>(0, 1).toDouble() / PROCESSING_SCALE;
        final x2 = line.at<int>(0, 2).toDouble() / PROCESSING_SCALE;
        final y2 = line.at<int>(0, 3).toDouble() / PROCESSING_SCALE;
        
        // Store actual points instead of just averages
        final points = [cv.Point(x1.toInt(), y1.toInt()), cv.Point(x2.toInt(), y2.toInt())];
        final avgX = (x1 + x2) / 2;
        final avgY = (y1 + y2) / 2;
        
        // More flexible angle filtering for curves
        final dy = (y2 - y1);
        final dx = (x2 - x1);
        final angle = dy != 0 ? (dx / dy).abs() : double.infinity;
        
        if (angle < 2.0 && avgY > height * 0.5) {  // More permissive angle threshold
          if (avgX < centerX) {
            _addToSegmentCluster(leftSegments, points, CLUSTER_DISTANCE);
          } else {
            _addToSegmentCluster(rightSegments, points, CLUSTER_DISTANCE);
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

        // Validate lane width
        final laneWidth = rightX - leftX;
        if (laneWidth > width * 0.2 && laneWidth < width * 0.9) {  // Expected lane width range
          laneCenter = (leftX + rightX) / 2;
          isValidLane = true;
        } else {
          laneCenter = centerX;
        }
      } else if (leftSegments.isNotEmpty) {
        double avgX = 0;
        for (var point in leftSegments.first) {
          avgX += point.x;
        }
        final leftX = avgX / leftSegments.first.length;
        laneCenter = leftX + (width * 0.35);
        isValidLane = leftSegments.first.length >= 3;
      } else if (rightSegments.isNotEmpty) {
        double avgX = 0;
        for (var point in rightSegments.first) {
          avgX += point.x;
        }
        final rightX = avgX / rightSegments.first.length;
        laneCenter = rightX - (width * 0.35);
        isValidLane = rightSegments.first.length >= 3;
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

      // Calculate confidence score based on number of segments
      double confidence = 0.0;
      if (leftSegments.isNotEmpty && rightSegments.isNotEmpty) {
        confidence = min(1.0, (leftSegments.first.length + rightSegments.first.length) / 10.0);
      } else if (leftSegments.isNotEmpty || rightSegments.isNotEmpty) {
        confidence = min(0.7, (leftSegments.isEmpty ? rightSegments.first.length : leftSegments.first.length) / 8.0);
      }

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
}