import pickle

import mysql.connector


def encode(result):
    return [[num[0], pickle.loads(num[1].encode('ISO-8859-1')), num[2] == 1] for num in result]


def decode(x):
    return x[0], pickle.dumps(x[1]).decode('ISO-8859-1'), True


class MysqlConnector:

    def __init__(self, host, user, password, database):
        self.database = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.database.cursor()

    def upload_dataset(self, users):
        for x in users:
            sql = "INSERT INTO users (name_surname, embedding, access) VALUES (%s, %s, %s)"
            val = decode(x)

            self.cursor.execute(sql, val)

        self.database.commit()
        print("import complete")

    def select_all_users(self):
        self.cursor.execute("SELECT * FROM users")
        return self.cursor.fetchall()
