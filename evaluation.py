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