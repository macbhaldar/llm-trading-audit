from .llm_reasoning import LLMReasoningAnalyzer
from .counterfactual import CounterfactualGenerator


class ExplainabilityPipeline:

    def __init__(
        self,
        shap_explainer=None,
        lime_explainer=None,
    ):

        self.shap = shap_explainer

        self.lime = lime_explainer

    def explain(

        self,

        model,

        X,

        reasoning,
    ):

        result = {}

        if self.shap:

            result["SHAP"] = self.shap.explain(X)

        if self.lime:

            result["LIME"] = self.lime.explain(

                X.iloc[0].values,

                model.predict,
            )

        result["Reasoning"] = (

            LLMReasoningAnalyzer.analyze(

                reasoning
            )

        )

        result["Counterfactual"] = (

            CounterfactualGenerator.generate(

                X.iloc[0]
            )

        )

        return result