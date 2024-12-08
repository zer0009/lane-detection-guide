import 'package:flutter/material.dart';

class LaneGuidePainter extends CustomPainter {
  final double deviation;

  LaneGuidePainter(this.deviation);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.green.withOpacity(0.5)
      ..strokeWidth = 3.0
      ..style = PaintingStyle.stroke;

    // Draw center line
    final centerX = size.width / 2;
    canvas.drawLine(
      Offset(centerX, size.height),
      Offset(centerX, size.height * 0.6),
      paint..color = Colors.white.withOpacity(0.5),
    );

    // Draw deviation line
    final deviationX = centerX + (deviation * centerX);
    canvas.drawLine(
      Offset(deviationX, size.height),
      Offset(deviationX, size.height * 0.6),
      paint..color = deviation.abs() < 0.1 ? Colors.green.withOpacity(0.5) : Colors.red.withOpacity(0.5),
    );

    // Draw guidance arrow
    if (deviation.abs() >= 0.1) {
      final arrowPaint = Paint()
        ..color = Colors.red.withOpacity(0.7)
        ..style = PaintingStyle.fill;

      final arrowPath = Path();
      final arrowY = size.height * 0.3;
      final arrowSize = 30.0;

      if (deviation > 0) {
        // Arrow pointing left
        arrowPath.moveTo(centerX - arrowSize, arrowY);
        arrowPath.lineTo(centerX + arrowSize, arrowY - arrowSize);
        arrowPath.lineTo(centerX + arrowSize, arrowY + arrowSize);
        arrowPath.close();
      } else {
        // Arrow pointing right
        arrowPath.moveTo(centerX + arrowSize, arrowY);
        arrowPath.lineTo(centerX - arrowSize, arrowY - arrowSize);
        arrowPath.lineTo(centerX - arrowSize, arrowY + arrowSize);
        arrowPath.close();
      }

      canvas.drawPath(arrowPath, arrowPaint);
    }
  }

  @override
  bool shouldRepaint(LaneGuidePainter oldDelegate) {
    return oldDelegate.deviation != deviation;
  }
} 