from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class WarehouseView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "WarehouseView",
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        # self._drawers = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/cabinet/drawer_top", name="drawers_view", reset_xform_properties=False
        # )
