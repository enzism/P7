import pickle
from sklearn.metrics import fbeta_score
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.custom import *


# Customiseing the explainerdashboard : each class refers to a sinlge Tab :
def custom_metric(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=3)

# First Tab : Inividual prediction
class Tab1(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Summary")
        self.performances = ConfusionMatrixComponent(explainer, hide_footer=True,
                                                     hide_popout=True, hide_selector=True)
        self.shap_summary = ShapSummaryComponent(explainer,
                                title='Impact',
                                hide_subtitle=True, hide_selector=True,
                                hide_depth=True, depth=15,
                                hide_cats=True, cats=True, hide_popout=True, hide_type=True)
        self.roc_curve = RocAucComponent(explainer, hide_cutoff=True,
                                hide_subtitle=True, hide_selector=True,
                                hide_cats=True, cats=True,
                                hide_index=True)

        self.metrics = ClassifierModelSummaryComponent(explainer, title='Model Metrics', show_metrics=['recall', 'precision', custom_metric], hide_cutoff=True,
                                                       hide_selector=True)
        self.connector = ShapSummaryDependenceConnector(
                self.shap_summary, self.roc_curve)

        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                     html.H3("Model Performance"),
                    html.Div("As you can see on the right, the model performs quite well."),
                    html.Div("The higher the predicted probability of survival predicted by "
                            "the model on the basis of learning from examples in the training set"
                            ", the higher is the actual percentage of passengers surviving in "
                            "the test set"),
                ]),
                    dbc.Row([
                        self.metrics.layout()
                    ], align='right')
                ], width=4, style=dict(margin=30)),
                dbc.Col([
                    self.performances.layout()
                ], style=dict(margin=30))
            ]),
            dbc.Row([
                dbc.Col([
                    self.shap_summary.layout()
                ], style=dict(margin=30)),
                dbc.Col([
                    html.H3("Feature Importances"),
                    html.Div("On the left you can check out for yourself which parameters were the most important."),
                    html.Div(f"Clearly {self.explainer.columns_ranked_by_shap()[0]} was the most important"
                            f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                            f" and {self.explainer.columns_ranked_by_shap()[2]}."),
                    html.Div("If you select 'detailed' you can see the impact of that variable on "
                            "each individual prediction. With 'aggregate' you see the average impact size "
                            "of that variable on the final prediction."),
                    html.Div("With the detailed view you can clearly see that the the large impact from Sex "
                            "stems both from males having a much lower chance of survival and females a much "
                            "higher chance.")
                ], width=4, style=dict(margin=30)),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Feature dependence"),
                    html.Div("In the plot to the right you can see that the higher the cost "
                            "of the fare that passengers paid, the higher the chance of survival. "
                            "Probably the people with more expensive tickets were in higher up cabins, "
                            "and were more likely to make it to a lifeboat."),
                    html.Div("When you color the impacts by PassengerClass, you can clearly see that "
                            "the more expensive tickets were mostly 1st class, and the cheaper tickets "
                            "mostly 3rd class."),
                    html.Div("On the right you can check out for yourself how different features impacted "
                            "the model output."),
                ], width=4, style=dict(margin=30)),
                dbc.Col([
                    self.roc_curve.layout()
                ], style=dict(margin=30)),
            ])
        ])


class Tab2(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Individual Prediction")

        self.index = ClassifierRandomIndexComponent(explainer,
                                                    hide_title=True, hide_index=False,
                                                    hide_slider=True, hide_labels=True,
                                                    hide_pred_or_perc=True,
                                                    hide_selector=True, hide_button=False)

        self.contributions = ShapContributionsGraphComponent(explainer, depth=10)

        self.prediction = ClassifierPredictionSummaryComponent(explainer, hide_title=True,
                                                               hide_index=True, hide_star_explanation=True)

        self.connector = IndexConnector(self.index, [self.contributions, self.prediction])

        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Prediction Probability"),
                    html.Div("As you can see on the right, the model performs quite well."),
                    html.Div("Text pour expliquer le diagrame que l'on observe"),
                ], width=4, style=dict(margin=30)),
                dbc.Col([
                    self.prediction.layout()
                ], style=dict(margin=30))
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Select Client ID :"),
                    self.index.layout()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Contributions to prediction:"),
                    self.contributions.layout()
                ])
            ])
        ])
model = pickle.load(open('trained_model', 'rb'))
data = pickle.load(open('test_data', 'rb'))
X_test, y_test = data

explainer = ClassifierExplainer(model, X_test, y_test)
db = ExplainerDashboard(explainer, [Tab1, Tab2], hide_header=True)
db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)