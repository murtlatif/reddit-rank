import cgi
import cgitb
from modeleval import evaluate_input
cgitb.enable()

form = cgi.FieldStorage()

title = form.getvalue('title')
nsfw = form.getvalue('nsfw')


print ("Content-type:text/html\r\n\r\n")
print ('<html>')
print ('    <head>')
print ('''    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
        integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

    <!-- Custom CSS -->
    <link type="text/css" rel="stylesheet" href="../css/main.css">''')

# print(f'        <meta http-equiv="refresh" content="0; URL=http://localhost:8080/">')
print ('        <title>RedditRank Evaluation</title>')
print ('    </head>')
print ('<body>')
print ('    <div class="container mt-3">')
print ('        <div class="row">')
print ('            <div class="col text-center">')
print ('                <h1 class="display-4">RedditRank Evaluation</h1>')
print ('            </div>')
print ('        </div>')
print ('        <div class="row">')
print ('            <div class="col">')
print ('                <a href="/">Go Back</a>')
print ('            </div>')
print ('        </div>')
print ('        <div class="row justify-content-center">')
print ('            <div class="col-md-auto text-center">')
print ('                <p>You have entered</p>')
print ('                <div class="pt-2 px-3" id="inputinfo">')
print(f'                    <p>Title: <em>{title}</em></p>')
print(f'                    <p>NSFW: <em>{nsfw}</em></p>')
print ('                </div>')
print(f'                <p>Your post score is <strong>{evaluate_input(title, nsfw)}</strong></p>')
print ('            </div>')
print ('        </div>')
print ('    </div>')
print ('</body>')
print ('</html>')