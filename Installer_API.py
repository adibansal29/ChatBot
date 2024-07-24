from flask import Flask,jsonify
import ndjson

# Create a Flask instance
app = Flask(__name__)

# Read the ndjson file
def read_file1(url):
    with open(url,'r', encoding='utf-8') as file1:
        data = ndjson.load(file1)
    return data

data_final=read_file1('data_2022-05-31T22_12_05.460216Z.ndjson')

@app.route("/")
def index():
    return 'Hi'


# Create a route to return the installer data
@app.route("/api/<zip_code>")
def ndjson_data(zip_code):
    count=0
    lst=[]
    while count!=3:
        for i in range(len(data_final)):
            if zip_code in data_final[i]["installations_by_zip"]:
                count+=1
                lst.append(i)
                if count==3:
                    break
        break
    final=[]
    for abc in lst:
        final.append(data_final[abc])


    return jsonify(final)

# Run the app
if __name__=="__main__":
    app.run(debug=True)