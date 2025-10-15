"""
Evaluation Metrics for VAD
Author: Kiro Chen (530337094)
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class VADEvaluator:
    """Evaluate VAD performance"""

    def __init__(self):
        pass

    def evaluate_frame_level(self, y_true, y_pred):
        """
        Frame-level evaluation metrics

        Args:
            y_true: Ground truth labels (n_frames,)
            y_pred: Predicted labels (n_frames,)

        Returns:
            metrics: Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1, zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'true_positive': int(tp),
            'false_positive': int(fp),
            'true_negative': int(tn),
            'false_negative': int(fn)
        }

        return metrics

    def evaluate_segment_level(self, true_segments, pred_segments, tolerance=0.5):
        """
        Segment-level evaluation with IoU (Intersection over Union)

        Args:
            true_segments: List of (label, start, end) tuples (ground truth)
            pred_segments: List of (label, start, end) tuples (predictions)
            tolerance: Time tolerance for matching segments (seconds)

        Returns:
            metrics: Dictionary of segment-level metrics
        """
        # Filter only vocal segments
        true_vocal = [(s, e) for l, s, e in true_segments if l == 'vocal']
        pred_vocal = [(s, e) for l, s, e in pred_segments if l == 'vocal']

        if len(true_vocal) == 0:
            return {'segment_precision': 0, 'segment_recall': 0, 'segment_f1': 0, 'mean_iou': 0}

        # Count matches
        matched_true = 0
        matched_pred = 0
        total_iou = 0

        for t_start, t_end in true_vocal:
            best_iou = 0
            for p_start, p_end in pred_vocal:
                iou = self._calculate_iou(t_start, t_end, p_start, p_end)
                if iou > best_iou:
                    best_iou = iou

            if best_iou > 0.5:  # Match if IoU > 0.5
                matched_true += 1
                total_iou += best_iou

        for p_start, p_end in pred_vocal:
            for t_start, t_end in true_vocal:
                iou = self._calculate_iou(t_start, t_end, p_start, p_end)
                if iou > 0.5:
                    matched_pred += 1
                    break

                    # Calculate metrics
                precision = matched_pred / len(pred_vocal) if len(pred_vocal) > 0 else 0
                recall = matched_true / len(true_vocal) if len(true_vocal) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                mean_iou = total_iou / len(true_vocal) if len(true_vocal) > 0 else 0

                metrics = {
                    'segment_precision': precision,
                    'segment_recall': recall,
                    'segment_f1': f1,
                    'mean_iou': mean_iou,
                    'true_segments': len(true_vocal),
                    'pred_segments': len(pred_vocal),
                    'matched_segments': matched_true
                }

                return metrics

            def _calculate_iou(self, start1, end1, start2, end2):
                """Calculate Intersection over Union for two time segments"""
                # Intersection
                intersection_start = max(start1, start2)
                intersection_end = min(end1, end2)
                intersection = max(0, intersection_end - intersection_start)

                # Union
                union_start = min(start1, start2)
                union_end = max(end1, end2)
                union = union_end - union_start

                # IoU
                iou = intersection / union if union > 0 else 0
                return iou

            def calculate_timing_errors(self, true_segments, pred_segments):
                """
                Calculate onset and offset timing errors

                Returns:
                    errors: Dictionary with timing statistics
                """
                true_vocal = [(s, e) for l, s, e in true_segments if l == 'vocal']
                pred_vocal = [(s, e) for l, s, e in pred_segments if l == 'vocal']

                onset_errors = []
                offset_errors = []

                for t_start, t_end in true_vocal:
                    # Find closest predicted segment
                    min_onset_error = float('inf')
                    min_offset_error = float('inf')

                    for p_start, p_end in pred_vocal:
                        onset_error = abs(p_start - t_start)
                        offset_error = abs(p_end - t_end)

                        if onset_error < min_onset_error:
                            min_onset_error = onset_error
                        if offset_error < min_offset_error:
                            min_offset_error = offset_error

                    if min_onset_error < float('inf'):
                        onset_errors.append(min_onset_error)
                    if min_offset_error < float('inf'):
                        offset_errors.append(min_offset_error)

                errors = {
                    'mean_onset_error': np.mean(onset_errors) if onset_errors else 0,
                    'mean_offset_error': np.mean(offset_errors) if offset_errors else 0,
                    'median_onset_error': np.median(onset_errors) if onset_errors else 0,
                    'median_offset_error': np.median(offset_errors) if offset_errors else 0
                }

                return errors

            def print_evaluation_report(self, metrics_frame, metrics_segment=None):
                """Print a comprehensive evaluation report"""
                print("\n" + "=" * 60)
                print("VOCAL ACTIVITY DETECTION - EVALUATION REPORT")
                print("=" * 60)

                print("\n--- Frame-Level Metrics ---")
                print(f"Accuracy:    {metrics_frame['accuracy']:.4f}")
                print(f"Precision:   {metrics_frame['precision']:.4f}")
                print(f"Recall:      {metrics_frame['recall']:.4f}")
                print(f"F1-Score:    {metrics_frame['f1_score']:.4f}")
                print(f"Specificity: {metrics_frame['specificity']:.4f}")

                print("\n--- Confusion Matrix ---")
                print(f"True Positives:  {metrics_frame['true_positive']}")
                print(f"False Positives: {metrics_frame['false_positive']}")
                print(f"True Negatives:  {metrics_frame['true_negative']}")
                print(f"False Negatives: {metrics_frame['false_negative']}")

                if metrics_segment:
                    print("\n--- Segment-Level Metrics ---")
                    print(f"Precision:   {metrics_segment['segment_precision']:.4f}")
                    print(f"Recall:      {metrics_segment['segment_recall']:.4f}")
                    print(f"F1-Score:    {metrics_segment['segment_f1']:.4f}")
                    print(f"Mean IoU:    {metrics_segment['mean_iou']:.4f}")
                    print(f"\nSegment Counts:")
                    print(f"  True segments:    {metrics_segment['true_segments']}")
                    print(f"  Pred segments:    {metrics_segment['pred_segments']}")
                    print(f"  Matched segments: {metrics_segment['matched_segments']}")

                print("\n" + "=" * 60)

            # Test the evaluator
            if __name__ == "__main__":
                print("Testing VAD Evaluator...")

                # Create synthetic ground truth and predictions
                # Ground truth: vocal at [0-3s, 5-8s, 10-12s]
                y_true = np.zeros(600)  # 600 frames = ~13.6s at hop=512, sr=22050
                y_true[0:132] = 1  # 0-3s
                y_true[220:352] = 1  # 5-8s
                y_true[440:528] = 1  # 10-12s

                # Predictions: similar but with some errors
                y_pred = y_true.copy()
                y_pred[10:30] = 0  # False negative
                y_pred[150:170] = 1  # False positive
                y_pred[450:460] = 0  # Small error

                # Create segments
                true_segments = [
                    ('vocal', 0.0, 3.0),
                    ('instrumental', 3.0, 5.0),
                    ('vocal', 5.0, 8.0),
                    ('instrumental', 8.0, 10.0),
                    ('vocal', 10.0, 12.0)
                ]

                pred_segments = [
                    ('vocal', 0.2, 2.8),  # Slightly off
                    ('instrumental', 2.8, 4.9),
                    ('vocal', 4.9, 8.1),  # Extended
                    ('instrumental', 8.1, 9.8),
                    ('vocal', 9.8, 12.2)  # Early start
                ]

                # Evaluate
                evaluator = VADEvaluator()

                metrics_frame = evaluator.evaluate_frame_level(y_true, y_pred)
                metrics_segment = evaluator.evaluate_segment_level(true_segments, pred_segments)
                timing_errors = evaluator.calculate_timing_errors(true_segments, pred_segments)

                # Print report
                evaluator.print_evaluation_report(metrics_frame, metrics_segment)

                print("\n--- Timing Errors ---")
                print(f"Mean onset error:   {timing_errors['mean_onset_error']:.3f}s")
                print(f"Mean offset error:  {timing_errors['mean_offset_error']:.3f}s")
                print(f"Median onset error: {timing_errors['median_onset_error']:.3f}s")
                print(f"Median offset error:{timing_errors['median_offset_error']:.3f}s")
