from utils.FaceDetector import FaceDetector
from utils.Loader import Loader
from utils.MysqlConnector import MysqlConnector

detector = FaceDetector()

mysql = MysqlConnector(host="localhost", user="python_user", password="password", database="python_project")

loader = Loader(dataset_path='D:/dataset')

users = []

users = loader.load(mtcnn=detector.mtcnn, resnet=detector.resnet, users=users)

mysql.upload_dataset(users)
