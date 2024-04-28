import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from rag import RAG

UPLOAD_FOLDER = "./pdf"
ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


rag = RAG()


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for("use_rag_app", name=filename))
    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """


@app.route("/use_rag_app", methods=["GET", "POST"])
def use_rag_app():
    if request.method == "POST":
        # Get the question from the form
        question = request.form["question"]

        file_name = request.args.get("name")
        # Generate response using RAG model

        response = rag.generate(file_name, question=question)

        # You may want to render the response in HTML and display it to the user
        return f"<h2>Response:</h2><p>{response}</p>"

    # If it's a GET request, show the form to input the question
    return """
    <!doctype html>
    <title>Ask a Question</title>
    <h1>Ask a Question</h1>
    <form method=post>
      <label for="question">Question:</label><br>
      <input type=text id="question" name="question"><br>
      <input type=submit value=Submit>
    </form>
    """
