from explainerdashboard import ExplainerDashboard
import lightgbm
db = ExplainerDashboard.from_config("dashboard.yaml")
app = db.flask_server()