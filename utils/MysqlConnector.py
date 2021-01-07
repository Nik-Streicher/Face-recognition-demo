import pickle

import mysql.connector


# encodes to bytes and deserialize and return database users table in format
# [[name, users embedding, access(True or False)], ...]
def encode(database_users):
    return [[num[0], pickle.loads(num[1].encode('ISO-8859-1')), num[2] == 1] for num in database_users]


# list format -> [name, embedding]
# serializes tensor object to bytes and decode to String
def decode(users_embedding):
    return pickle.dumps(users_embedding).decode('ISO-8859-1')


class MysqlConnector:

    # initialize database and makes cursor
    def __init__(self, host, user, password, database):
        self.database = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.database.cursor()

    # Uploads users list to database.
    # List format -> [name, Tensor object]
    def upload_dataset(self, users):
        for x in users:
            if self.find_duplicates(x[1]):
                pass
            else:
                sql = "INSERT INTO users (name_surname, embedding, access) VALUES (%s, %s, %s)"
                val = x[0], decode(users_embedding=x[1]), True

                self.cursor.execute(sql, val)

        self.database.commit()
        print("import complete")

    # selects all columns from table "users" and return them in format
    # [name, String representation of bytes object, access(1 == True; 2 == False)]
    def select_all_users(self):
        self.cursor.execute("SELECT * FROM users")
        return self.cursor.fetchall()

    # looks for duplicates
    def find_duplicates(self, inserted_embedding):
        flag = False

        for y in encode(self.select_all_users()):
            if (inserted_embedding - y[1]).norm().item() == 0:
                flag = True

        return flag
