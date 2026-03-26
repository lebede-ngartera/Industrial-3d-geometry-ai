"""Design anomaly detection workflow.

'Detect unusual geometric patterns that may indicate manufacturability
or quality risks.' — links to industrial AI and production quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from geofusion.models.anomaly import GeometryAnomalyDetector

logger = logging.getLogger(__name__)


@dataclass
class AnomalyReport:
    """Report from anomaly detection analysis."""

    part_id: str
    is_anomalous: bool
    anomaly_score: float
    threshold: float
    risk_level: str  # "normal", "warning", "critical"
    details: dict = field(default_factory=dict)


class AnomalyDetectionWorkflow:
    """Engineering workflow for detecting geometric anomalies.

    Identifies unusual geometric patterns that may indicate:
    - Manufacturability issues
    - Quality risks
    - Design errors
    - Non-standard features requiring review

    Workflow:
    1. Train on known-good parts (normal distribution)
    2. Set threshold from validation data
    3. Score new designs for anomaly risk
    """

    def __init__(
        self,
        detector: GeometryAnomalyDetector,
        device: str = "cuda",
        warning_percentile: float = 90.0,
        critical_percentile: float = 99.0,
    ):
        self.detector = detector.to(device)
        self.device = device
        self.warning_threshold: float | None = None
        self.critical_threshold: float | None = None
        self.warning_percentile = warning_percentile
        self.critical_percentile = critical_percentile

    def calibrate(self, normal_dataloader: DataLoader) -> dict[str, float]:
        """Calibrate thresholds using known-good parts.

        Args:
            normal_dataloader: DataLoader of normal (non-anomalous) parts

        Returns:
            Dictionary with computed thresholds
        """
        self.detector.eval()
        all_scores = []

        with torch.no_grad():
            for batch in normal_dataloader:
                points = batch["points"].to(self.device)
                scores = self.detector.anomaly_score(points)
                all_scores.append(scores.cpu())

        scores_np = torch.cat(all_scores).numpy()

        self.warning_threshold = float(np.percentile(scores_np, self.warning_percentile))
        self.critical_threshold = float(np.percentile(scores_np, self.critical_percentile))
        self.detector.threshold = self.warning_threshold

        logger.info(
            f"Calibrated thresholds — Warning: {self.warning_threshold:.6f}, "
            f"Critical: {self.critical_threshold:.6f}"
        )

        return {
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "mean_normal_score": float(scores_np.mean()),
            "std_normal_score": float(scores_np.std()),
        }

    @torch.no_grad()
    def analyze(
        self,
        points: torch.Tensor,
        part_id: str = "unknown",
    ) -> AnomalyReport:
        """Analyze a single part for anomalies.

        Args:
            points: (N, C) or (1, N, C) point cloud
            part_id: Part identifier

        Returns:
            AnomalyReport with risk assessment
        """
        if self.warning_threshold is None:
            raise RuntimeError("Call calibrate() before analyze()")

        if points.ndim == 2:
            points = points.unsqueeze(0)
        points = points.to(self.device)

        self.detector.eval()
        score = self.detector.anomaly_score(points).item()

        # Determine risk level
        if score > self.critical_threshold:
            risk_level = "critical"
            is_anomalous = True
        elif score > self.warning_threshold:
            risk_level = "warning"
            is_anomalous = True
        else:
            risk_level = "normal"
            is_anomalous = False

        return AnomalyReport(
            part_id=part_id,
            is_anomalous=is_anomalous,
            anomaly_score=score,
            threshold=self.warning_threshold,
            risk_level=risk_level,
            details={
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold,
                "score_vs_warning": score / self.warning_threshold,
            },
        )

    @torch.no_grad()
    def batch_analyze(
        self,
        dataloader: DataLoader,
    ) -> list[AnomalyReport]:
        """Analyze a batch of parts for anomalies.

        Args:
            dataloader: DataLoader of parts to analyze

        Returns:
            List of AnomalyReport
        """
        reports = []

        for batch in dataloader:
            points = batch["points"].to(self.device)
            scores = self.detector.anomaly_score(points)

            for i in range(points.shape[0]):
                part_id = batch.get("filename", [f"part_{i}"] * points.shape[0])
                if isinstance(part_id, (list, tuple)):
                    pid = part_id[i]
                else:
                    pid = str(part_id[i])

                score = scores[i].item()

                if score > self.critical_threshold:
                    risk_level = "critical"
                    is_anomalous = True
                elif score > self.warning_threshold:
                    risk_level = "warning"
                    is_anomalous = True
                else:
                    risk_level = "normal"
                    is_anomalous = False

                reports.append(
                    AnomalyReport(
                        part_id=pid,
                        is_anomalous=is_anomalous,
                        anomaly_score=score,
                        threshold=self.warning_threshold,
                        risk_level=risk_level,
                    )
                )

        logger.info(
            f"Analyzed {len(reports)} parts — "
            f"{sum(1 for r in reports if r.risk_level == 'critical')} critical, "
            f"{sum(1 for r in reports if r.risk_level == 'warning')} warning, "
            f"{sum(1 for r in reports if r.risk_level == 'normal')} normal"
        )

        return reports
