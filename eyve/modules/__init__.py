"""
Eyve inspection modules — production-time analysis beyond detection.

Available modules are listed in MODULE_REGISTRY; the production screen
builds its "Módulos" section from it, so adding a new module (counting,
measurement, edges, assembly…) means: implement InspectionModule in a new
file and register it here.
"""
from eyve.modules.base import InspectionModule, ModuleVerdict
from eyve.modules.polarity_module import PolarityModule
from eyve.modules.counting_module import CountingModule

#: name → class of every available module
MODULE_REGISTRY = {
    PolarityModule.name: PolarityModule,
    CountingModule.name: CountingModule,
}

__all__ = ["InspectionModule", "ModuleVerdict", "PolarityModule",
           "CountingModule", "MODULE_REGISTRY"]
