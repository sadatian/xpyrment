"""Personalization, Uplift Modeling, and Heterogeneous Treatment Effects (HTE).

Submodules:
- `meta_learners`: Implements S-Learner, T-Learner, and X-Learner.
- `causal_forest`: Implements Causal Trees and Causal Forests.
"""

from xpyrment.personalize.meta_learners import SLearner, TLearner, XLearner, ElasticNetRegressor
from xpyrment.personalize.causal_forest import CausalTree, CausalForest
from xpyrment.personalize.double_ml import DoubleMachineLearning
from xpyrment.personalize.dtr import QFactorModel, DynamicTreatmentRegime
from xpyrment.personalize.infinite_mixture import InfiniteDirichletClusterer
from xpyrment.personalize.subgroup import SubgroupHeterogeneityDiscoverer

__all__ = [
    "SLearner",
    "TLearner",
    "XLearner",
    "ElasticNetRegressor",
    "CausalTree",
    "CausalForest",
    "DoubleMachineLearning",
    "QFactorModel",
    "DynamicTreatmentRegime",
    "InfiniteDirichletClusterer",
    "SubgroupHeterogeneityDiscoverer",
]
