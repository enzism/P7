from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.custom import *
import lightgbm

db = ExplainerDashboard.from_config("dashboard.yaml")
app = db.flask_server()


