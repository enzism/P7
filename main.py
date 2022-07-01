import pickle

from explainerdashboard import ClassifierExplainer, ExplainerDashboard

model = pickle.load(open('trained_model', 'rb'))
data = pickle.load(open('test_data', 'rb'))
X_test = data

explainer = ClassifierExplainer(model, X_test) #, y_test)
# ExplainerDashboard(explainer).run()
# ExplainerDashboard à réadapter
db = ExplainerDashboard(explainer, title="Cool Title", shap_interaction=False)
db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)