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
      final blurred = cv.gaussianBlur(gray, (5, 5), 1.5);  // Increased blur
      resources.add(blurred);

      // Simplified edge detection with fixed thresholds for speed
      final edges = cv.canny(blurred, 30, 90);  // Adjusted thresholds
      resources.add(edges);

      // Apply ROI mask
      final mask = _createROIMask(processWidth, processHeight);
      final maskedEdges = cv.Mat.zeros(processHeight, processWidth, cv.MatType.CV_8UC1);
      resources.add(maskedEdges);
      cv.bitwiseAND(edges, mask, dst: maskedEdges);

      // Optimize Hough transform parameters
      final lines = cv.HoughLinesP(
        maskedEdges,
        1.0,  // Reduced step size for better accuracy
        pi / 180.0,
        20,   // Increased threshold
        minLineLength: 30.0,  // Longer minimum line length
        maxLineGap: 50.0      // Reduced max gap
      );
      resources.add(lines);

      if (lines.rows < 2) {
        _lastResult = LaneDetectionResult(0.0, imageBytes, false);
        return _lastResult!;
      }

      // Enhanced line filtering and clustering
      final List<List<double>> leftClusters = [];
      final List<List<double>> rightClusters = [];
      const double CLUSTER_THRESHOLD = 50.0; // pixels
      
      for (int i = 0; i < lines.rows; i++) {
        final line = lines.row(i);
        final x1 = line.at<int>(0, 0).toDouble() / PROCESSING_SCALE;
        final y1 = line.at<int>(0, 1).toDouble() / PROCESSING_SCALE;
        final x2 = line.at<int>(0, 2).toDouble() / PROCESSING_SCALE;
        final y2 = line.at<int>(0, 3).toDouble() / PROCESSING_SCALE;
        
        final dy = (y2 - y1);
        final dx = (x2 - x1);
        final angle = dy != 0 ? (dx / dy).abs() : double.infinity;
        
        // Stricter angle filtering
        if (angle > 0.3 && angle < 1.2) {  // Even narrower angle range
          final avgX = (x1 + x2) / 2;
          final avgY = (y1 + y2) / 2;
          final lineLength = sqrt(dx * dx + dy * dy);
          
          // Only consider longer lines in the lower portion
          if (avgY > height * 0.6 && lineLength > 30) {
            bool addedToCluster = false;
            
            if (avgX < centerX) {
              // Try to add to existing left clusters
              for (var cluster in leftClusters) {
                if ((cluster.reduce((a, b) => a + b) / cluster.length - avgX).abs() < CLUSTER_THRESHOLD) {
                  cluster.add(avgX);
                  addedToCluster = true;
                  break;
                }
              }
              if (!addedToCluster) {
                leftClusters.add([avgX]);
              }
            } else {
              // Try to add to existing right clusters
              for (var cluster in rightClusters) {
                if ((cluster.reduce((a, b) => a + b) / cluster.length - avgX).abs() < CLUSTER_THRESHOLD) {
                  cluster.add(avgX);
                  addedToCluster = true;
                  break;
                }
              }
              if (!addedToCluster) {
                rightClusters.add([avgX]);
              }
            }

            cv.line(
              visualOutput,
              cv.Point(x1.toInt(), y1.toInt()),
              cv.Point(x2.toInt(), y2.toInt()),
              avgX < centerX ? cv.Scalar(255, 0, 0) : cv.Scalar(0, 0, 255),
              thickness: 2
            );
          }
        }
      }

      // Filter out small clusters and calculate lane center
      leftClusters.removeWhere((cluster) => cluster.length < 2);
      rightClusters.removeWhere((cluster) => cluster.length < 2);

      double laneCenter;
      bool isValidLane = false;

      if (leftClusters.isNotEmpty && rightClusters.isNotEmpty) {
        // Sort clusters by size and average position
        leftClusters.sort((a, b) => b.length.compareTo(a.length));
        rightClusters.sort((a, b) => b.length.compareTo(a.length));

        // Use the largest clusters
        final leftX = leftClusters.first.reduce((a, b) => a + b) / leftClusters.first.length;
        final rightX = rightClusters.first.reduce((a, b) => a + b) / rightClusters.first.length;

        // Validate lane width
        final laneWidth = rightX - leftX;
        if (laneWidth > width * 0.2 && laneWidth < width * 0.9) {  // Expected lane width range
          laneCenter = (leftX + rightX) / 2;
          isValidLane = true;
        } else {
          laneCenter = centerX;
        }
      } else if (leftClusters.isNotEmpty) {
        final leftX = leftClusters.first.reduce((a, b) => a + b) / leftClusters.first.length;
        laneCenter = leftX + (width * 0.35);
        isValidLane = leftClusters.first.length >= 3;  // Require more points for single-line detection
      } else if (rightClusters.isNotEmpty) {
        final rightX = rightClusters.first.reduce((a, b) => a + b) / rightClusters.first.length;
        laneCenter = rightX - (width * 0.35);
        isValidLane = rightClusters.first.length >= 3;  // Require more points for single-line detection
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

      _lastResult = LaneDetectionResult(
        deviation,
        cv.imencode('.jpg', visualOutput).$2,
        isValidLane
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
  }
}