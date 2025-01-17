from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import LR_with_enc

#Executable code for the loan approval prediction application

app = Flask(__name__)

df = pd.read_csv("./Dataset.csv")
df.dropna(inplace=True)
df.drop('Loan_ID', axis=1, inplace=True)
df_cat = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']]
df_num = df[['ApplicantIncome',	'CoapplicantIncome',	'LoanAmount',	'Loan_Amount_Term',	'Credit_History']]
df_dummies = pd.get_dummies(df_cat)
df_dummies = df_dummies.drop(['Gender_Male', 'Married_No', 'Education_Not Graduate', 'Self_Employed_No'], axis=1)
df1 = pd.concat([df_dummies, df_num], axis=1)

# import the model and train once before running the service
model = LR_with_enc.LR_enc()
model.enc_setting()
model.training()


# Preprocess the categorical and numerical features of the user input data
def preprocess(df):
    df_cat = df[['Gender_Female', 'Married_Yes', 'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Graduate', 'Self_Employed_Yes', 'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']]
    df_num = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]
    df = pd.concat([df_cat, df_num], axis=1)
    df = pd.concat([df, df1], axis=0)

    df_cat = df[['Gender_Female', 'Married_Yes', 'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+','Education_Graduate', 'Self_Employed_Yes', 'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']]
    df_num = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]

    scaler = StandardScaler()
    scaler.fit(df_num)
    num_scaled = scaler.transform(df_num)
    df_scaled = pd.DataFrame(num_scaled)
    df_scaled.index = df_num.index
    df_scaled.columns = df_num.columns
    df_trans = pd.concat([df_cat, df_scaled], axis=1)
    df_trans = pd.DataFrame([df_trans.iloc[0]])
    return df_trans

# Use rendering to turn your code into an interactive page that users see when they visit your website,
# showing the screen of a form where users can enter personal information to predict whether they will be approved for a loan.
@app.route('/')
def main():
    return render_template('main.html')

# Executed when a client sends a POST method request to the '/result3' path
@app.route('/result3', methods=['POST'])
def result3():
    # Save the personal information entered by the user in the appropriate data type
    name = request.form['uname']
    gender = int(request.form['gender'])
    mrg = int(request.form['mrg'])
    edu = int(request.form['edu'])
    self = int(request.form['self'])
    dep = request.form['dep']
    income = int(request.form['income'])
    co_income = int(request.form['co_income'])
    credit = int(request.form['credit'])
    prop = request.form['prop']
    amount = int(request.form['amount'])
    term = int(request.form['term'])

    # Make a list of the stored values. In this case, all values selected by the user other than those entered directly are zeroed.
    list = [gender, mrg, 0, 0, 0, 0, edu, self, income, co_income, amount, term, credit, 0, 0, 0]

    # Based on the value selected by the user, change only that value from 0 to 1 among several values
    if dep == "0":
        list[2] = 1
    elif dep == "1":
        list[3] = 1
    elif dep == "2":
        list[4] = 1
    elif dep == "3+":
        list[5] = 1

    if prop == "Urban":
        list[13] = 1
    elif prop == "Semiurban":
        list[14] = 1
    elif prop == "Rural":
        list[15] = 1

    # To preprocess the user input data, convert the data into dataframe type
    df = pd.DataFrame(data=[list], columns=['Gender_Female', 'Married_Yes', 'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Graduate', 'Self_Employed_Yes', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area_Urban','Property_Area_Semiurban','Property_Area_Rural'])
    df = preprocess(df)

    arr = df.values.tolist() # Convert the preprocessed data back to an array format for the model to predict whether the loan will be approved.
    pred = model.predict(arr[0]) # Loan approval prediction results
    return render_template('result.html', data=pred, user=name) # pass in the prediction and the user's name, and render the result page


if __name__ == '__main__':
    app.run(debug=True)
