import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'dart:typed_data';
import 'dart:math' show pi, min, max;
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
      cv.Point((width * 0.65).toInt(), (height * 0.6).toInt()),  
      cv.Point((width * 0.35).toInt(), (height * 0.6).toInt()),  
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
      final blurred = cv.gaussianBlur(gray, (3, 3), 0);  // Smaller kernel
      resources.add(blurred);

      // Simplified edge detection with fixed thresholds for speed
      final edges = cv.canny(blurred, 50, 150);
      resources.add(edges);

      // Apply ROI mask
      final mask = _createROIMask(processWidth, processHeight);
      final maskedEdges = cv.Mat.zeros(processHeight, processWidth, cv.MatType.CV_8UC1);
      resources.add(maskedEdges);
      cv.bitwiseAND(edges, mask, dst: maskedEdges);

      // Optimize Hough transform parameters
      final lines = cv.HoughLinesP(
        maskedEdges,
        2.0,  // Increase step size
        pi / 180.0,
        15,   // Lower threshold
        minLineLength: 15.0,  // Shorter lines
        maxLineGap: 100.0     // Larger gaps allowed
      );
      resources.add(lines);

      if (lines.rows < 2) {
        _lastResult = LaneDetectionResult(0.0, imageBytes, false);
        return _lastResult!;
      }

      // Efficient line filtering
      final centerX = width / 2;
      List<double> leftLines = [];
      List<double> rightLines = [];

      for (int i = 0; i < lines.rows; i++) {
        final line = lines.row(i);
        final x1 = line.at<int>(0, 0).toDouble() / PROCESSING_SCALE;
        final y1 = line.at<int>(0, 1).toDouble() / PROCESSING_SCALE;
        final x2 = line.at<int>(0, 2).toDouble() / PROCESSING_SCALE;
        final y2 = line.at<int>(0, 3).toDouble() / PROCESSING_SCALE;
        
        final angle = (y2 - y1) != 0 ? ((x2 - x1) / (y2 - y1)).abs() : double.infinity;
        if (angle > 0.2 && angle < 2.0) {
          final avgX = (x1 + x2) / 2;
          
          // Draw lines only if they're significantly different from previous frame
          cv.line(
            visualOutput,
            cv.Point(x1.toInt(), y1.toInt()),
            cv.Point(x2.toInt(), y2.toInt()),
            avgX < centerX ? cv.Scalar(255, 0, 0) : cv.Scalar(0, 0, 255),
            thickness: 2
          );

          if (avgX < centerX) {
            leftLines.add(avgX);
          } else {
            rightLines.add(avgX);
          }
        }
      }

      // Quick early return if no valid lines
      if (leftLines.isEmpty && rightLines.isEmpty) {
        _lastResult = LaneDetectionResult(0.0, cv.imencode('.jpg', visualOutput).$2, false);
        return _lastResult!;
      }

      // Efficient lane center calculation
      double laneCenter;
      if (leftLines.isNotEmpty && rightLines.isNotEmpty) {
        leftLines.sort();
        rightLines.sort();
        final leftMedian = leftLines[leftLines.length ~/ 2];
        final rightMedian = rightLines[rightLines.length ~/ 2];
        laneCenter = (leftMedian + rightMedian) / 2;
      } else if (leftLines.isNotEmpty) {
        laneCenter = leftLines[leftLines.length ~/ 2] + width * 0.25;
      } else {
        laneCenter = rightLines[rightLines.length ~/ 2] - width * 0.25;
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

      // Calculate deviation
      double deviation = (laneCenter - centerX) / (width / 2);
      deviation = deviation.clamp(-1.0, 1.0);

      _lastResult = LaneDetectionResult(
        deviation,
        cv.imencode('.jpg', visualOutput).$2,
        true
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