from flask import Flask,render_template,request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/",methods=["POST"])
def hello():
    sen1=request.form["Sentence1"]
    sen2=request.form["Sentence2"]
    sentences = [sen1, sen2]

    model_name = 'all-MiniLM-L6-v2'

    model = SentenceTransformer(f'sentence-transformers/{model_name}')

    embeddings = model.encode(sentences)
    print(embeddings)
    similarity_percentage = cosine_similarity([embeddings[0]], [embeddings[1]])

    # return f"this is the first sentence {sen1},this is the second sentence {sen2}"
    return render_template('index.html',out = round(similarity_percentage[0][0],2))

if __name__=="__main__":
    app.run(debug=True)