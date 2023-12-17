from flask import Flask, redirect, url_for, request, render_template

app = Flask(__name__)

@app.route('/')
def hello_world():
   return "Hello World"

@app.route('/home')
def home():
   return render_template('home.html')

@app.route('/hello/<name>')
def hello_name(name):
   return "Hello World %s!" % name

@app.route('/blog/<int:postID>')
def show_blog(postID):
   return "Blog %d" % postID 

@app.route('/rev/<float:revNo>')
def revision(revNo):
   return 'Revision Number %f' % revNo

@app.route('/admin')
def hello_admin():
   return 'Hello Admin'

@app.route('/guest/<guest>')
def hello_guest(guest):
   return 'Hello %s as Guest' % guest

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route("/javascript")
def javascript():
   return render_template("javascript.html")


@app.route('/result')
def result():
   dict = {'phy':50,'che':60,'maths':70}
   return render_template('result.html', result = dict)

@app.route('/hello2/<int:score>')
def hello2_name(score):
   return render_template('hello.html', marks = score)



@app.route('/login', methods=['POST', 'GET'])
def login():
   if(request.method == 'POST'):
       user = request.form['nm']
       return redirect(url_for('success', name=user)) 
   else:
      user=request.args.get('nm')
      return redirect(url_for('success',name = user))


@app.route('/user/<name>')
def hello_user(name):
   if name =='admin':
      return redirect(url_for('hello_admin'))
   else:
      return redirect(url_for('hello_guest',guest = name))
if __name__ == '__main__':
 app.run(debug = True)