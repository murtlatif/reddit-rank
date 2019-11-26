import cgi
import cgitb
from modeleval import evaluate_input
cgitb.enable()

form = cgi.FieldStorage()

title = form.getvalue('title')
nsfw = form.getvalue('nsfw')
classifier_type = form.getvalue('classifier_type')

classifiers = {
    '2': {
        'possible_outcomes': ['low', 'high'], 
        'outcome_conversions': ['score is 100 or less', 'score is higher than 100']
    },
    '3': {
        'possible_outcomes': ['very low', 'okay', 'high'], 
        'outcome_conversions': ['score is 0 or 1', 'score is between 2 and 499', 'score is 500 or higher']
    },
    '5': {
        'possible_outcomes': ['zero', 'low', 'decent', 'high', 'viral'], 
        'outcome_conversions': ['score is 0', 'score is between 1 and 9', 'score is between 10 and 99', 'score is between 100 and 999', 'score is 1000 or higher']
    }
}

prediction_data = evaluate_input(title, nsfw, classifier_type)

print("Content-type:text/html\r\n\r\n")
print('<html>')
print('    <head>')
print('''    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
        integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

    <!-- Custom CSS -->
    <link type="text/css" rel="stylesheet" href="../css/main.css">''')

# print(f'        <meta http-equiv="refresh" content="0; URL=http://localhost:8080/">')
print('        <title>RedditRank Evaluation</title>')
print('    </head>')
print('<body>')
print('    <div class="container mt-3">')
# print('        <div class="row">')
# print('            <div class="col text-center">')
# print('                <h1 class="display-4">RedditRank Evaluation</h1>')
# print('            </div>')
# print('        </div>')
print('        <div class="row">')
print('            <div class="col">')
print('                <a href="/">Go Back</a>')
print('            </div>')
print('        </div>')
print('        <div class="row justify-content-center">')
print('            <div class="col-md-6 text-center">')
# print('                <p>You have entered</p>')
# print('                <div class="pt-2 px-3" id="inputinfo">')
# for data_item in prediction_data:
#     print(
#         f'                    <p class="mt-0">{data_item} -> <em>{prediction_data[data_item]}</em></p>')
# # print(f'                    <p>{prediction_data}</p>')
# # print(f'                    <p>Title: <em>{title}</em></p>')
# # print(f'                    <p>NSFW: <em>{nsfw}</em></p>')
# print('                </div>')
print(
    f'                <h1 class="display-3">Your post score is <strong>{classifiers[classifier_type]["possible_outcomes"][prediction_data["prediction"]]}</strong></h1>')
print('            </div>')
print('         </div>')
print('         <div class="row mt-5 justify-content-center">')
print('             <div class="col-6 mt-5">')
print('             <table class="table">')
print('                <tbody>')
for i in range(int(classifier_type)):
    print('                        <tr>')
    print('                            <td>')
    print(f'                                <strong style="font-size:30px">{classifiers[classifier_type]["possible_outcomes"][i]}</strong>')
    print('                            </td>')
    print('                            <td>')
    print(f'                                <span style="font-size:30px">{classifiers[classifier_type]["outcome_conversions"][i]}</span>')
    print('                            </td>')
    print('                        </tr>')
print('                    </tbody>')
print('                </table>')
print('            </div>')
print('        </div>')
print('    </div>')
print('</body>')
print('</html>')
