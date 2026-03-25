"""Synthetic text metadata generation for 3D shapes.

Generates engineering-style text descriptions for 3D objects to enable
multimodal learning without requiring manually annotated text data.
"""

from __future__ import annotations

import numpy as np


# Engineering-style description templates per category
TEMPLATES = {
    "airplane": [
        "A {adj} aircraft component with {wing} wings and {detail}.",
        "Aerodynamic {adj} airframe structure featuring {detail}.",
        "{adj} fuselage assembly with {wing} wing configuration and {detail}.",
    ],
    "car": [
        "A {adj} automotive body shell with {detail}.",
        "{adj} vehicle chassis structure featuring {detail}.",
        "Automotive {adj} exterior panel with {detail}.",
    ],
    "chair": [
        "A {adj} seating structure with {detail}.",
        "{adj} ergonomic chair frame with {detail}.",
        "Structural {adj} seat assembly featuring {detail}.",
    ],
    "table": [
        "A {adj} planar support structure with {detail}.",
        "{adj} horizontal work surface with {detail}.",
        "Structural {adj} table frame with {detail}.",
    ],
    "lamp": [
        "A {adj} lighting fixture with {detail}.",
        "{adj} illumination device featuring {detail}.",
        "Structural {adj} lamp assembly with {detail}.",
    ],
    "guitar": [
        "A {adj} stringed instrument body with {detail}.",
        "{adj} acoustic resonance chamber with {detail}.",
    ],
    "laptop": [
        "A {adj} portable computing enclosure with {detail}.",
        "{adj} hinged display-keyboard assembly with {detail}.",
    ],
    "mug": [
        "A {adj} cylindrical vessel with handle attachment and {detail}.",
        "{adj} beverage container with {detail}.",
    ],
    "knife": [
        "A {adj} blade component with {detail}.",
        "{adj} cutting tool assembly with {detail}.",
    ],
    "pistol": [
        "A {adj} handheld mechanical assembly with {detail}.",
        "{adj} grip-barrel mechanism with {detail}.",
    ],
}

DEFAULT_TEMPLATES = [
    "A {adj} 3D component with {detail}.",
    "{adj} engineering part featuring {detail}.",
    "Structural {adj} geometric assembly with {detail}.",
    "{adj} industrial component with {detail}.",
]

ADJECTIVES = [
    "compact", "elongated", "symmetric", "asymmetric", "angular",
    "curved", "lightweight", "robust", "streamlined", "modular",
    "thin-walled", "solid", "hollow", "reinforced", "tapered",
]

DETAILS = [
    "smooth surface finish",
    "complex curvature profiles",
    "multiple attachment points",
    "load-bearing geometry",
    "interlocking features",
    "precision-machined interfaces",
    "optimized material distribution",
    "thermal management features",
    "vibration dampening elements",
    "aerodynamic contour lines",
    "structural ribbing patterns",
    "uniform wall thickness",
    "draft angles for manufacturability",
    "snap-fit connection points",
    "weight-reducing through-holes",
]

WING_TYPES = ["swept", "delta", "straight", "tapered", "forward-swept"]

MANUFACTURING_PROPERTIES = [
    "injection molded", "CNC machined", "3D printed", "cast", "forged",
    "sheet metal formed", "extruded", "stamped", "laser cut", "die cast",
]

MATERIALS = [
    "aluminum alloy", "steel", "titanium", "carbon fiber composite",
    "ABS plastic", "nylon", "polycarbonate", "stainless steel",
    "magnesium alloy", "glass-filled polymer",
]


class TextMetadataGenerator:
    """Generate synthetic engineering text descriptions for 3D shapes.

    Produces varied, engineering-style text descriptions based on category
    and geometric properties of point clouds. This enables multimodal
    training without manual text annotation.
    """

    def __init__(self, seed: int = 42, include_properties: bool = True):
        self.rng = np.random.RandomState(seed)
        self.include_properties = include_properties

    def generate(
        self,
        category: str,
        points: np.ndarray | None = None,
        model_id: str | None = None,
    ) -> str:
        """Generate a text description for a 3D shape.

        Args:
            category: Shape category name.
            points: Point cloud array (N, 3+).
            model_id: Unique model identifier for reproducibility.

        Returns:
            Engineering-style text description.
        """
        # Select template
        templates = TEMPLATES.get(category, DEFAULT_TEMPLATES)
        template = templates[self.rng.randint(len(templates))]

        # Fill template
        adj = ADJECTIVES[self.rng.randint(len(ADJECTIVES))]
        detail = DETAILS[self.rng.randint(len(DETAILS))]
        wing = WING_TYPES[self.rng.randint(len(WING_TYPES))]

        text = template.format(adj=adj, detail=detail, wing=wing)

        # Add geometric properties if point cloud available
        if points is not None and self.include_properties:
            geo_desc = self._describe_geometry(points)
            text = f"{text} {geo_desc}"

        # Add manufacturing context
        if self.include_properties:
            mfg = MANUFACTURING_PROPERTIES[self.rng.randint(len(MANUFACTURING_PROPERTIES))]
            mat = MATERIALS[self.rng.randint(len(MATERIALS))]
            text = f"{text} Suitable for {mfg} in {mat}."

        return text

    def _describe_geometry(self, points: np.ndarray) -> str:
        """Generate geometric property description from point cloud."""
        xyz = points[:, :3]

        # Bounding box dimensions
        bb_min = xyz.min(axis=0)
        bb_max = xyz.max(axis=0)
        dims = bb_max - bb_min
        aspect_ratio = dims.max() / (dims.min() + 1e-8)

        # Volume approximation (convex hull-ish)
        volume_approx = np.prod(dims)

        # Surface area approximation
        center = xyz.mean(axis=0)
        radii = np.linalg.norm(xyz - center, axis=1)
        mean_radius = radii.mean()
        std_radius = radii.std()

        # Symmetry approximation
        reflected = xyz.copy()
        reflected[:, 0] = -reflected[:, 0]
        # Simple symmetry score based on distance distribution
        sym_score = 1.0 - min(std_radius / (mean_radius + 1e-8), 1.0)

        parts = []
        if aspect_ratio > 3:
            parts.append("highly elongated profile")
        elif aspect_ratio > 1.5:
            parts.append("moderately elongated form")
        else:
            parts.append("roughly equidimensional shape")

        if sym_score > 0.7:
            parts.append("high geometric symmetry")
        elif sym_score > 0.4:
            parts.append("moderate symmetry")

        parts.append(
            f"bounding dimensions {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f}"
        )

        return "Exhibits " + ", ".join(parts) + "."

    def generate_batch(
        self,
        categories: list[str],
        points_batch: list[np.ndarray] | None = None,
    ) -> list[str]:
        """Generate text descriptions for a batch of shapes."""
        texts = []
        for i, cat in enumerate(categories):
            pts = points_batch[i] if points_batch else None
            texts.append(self.generate(category=cat, points=pts))
        return texts
