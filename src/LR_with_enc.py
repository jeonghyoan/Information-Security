import piheaan as heaan
from piheaan.math import approx
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Logistic Regression Model code to predict whether a loan will be approved
# Turned the model into a class so that the service can use the trained model
class LR_enc:
    def __init__(self, eval=None, betas=None, context=None, dec=None, enc=None, sk=None, key_generator=None,
                 log_slots=15, num_slots=2 ** 15):
        self.eval = eval
        self.betas = betas
        self.context = context
        self.dec = dec
        self.enc = enc
        self.sk = sk
        self.key_generator = key_generator
        self.log_slots = log_slots
        self.num_slots = num_slots

    def enc_setting(self):
        # set parameter
        params = heaan.ParameterPreset.FGb
        self.context = heaan.make_context(params)  # context has paramter information
        heaan.make_bootstrappable(self.context)  # make parameter bootstrapable

        # create and save keys
        key_file_path = "keys"
        self.sk = heaan.SecretKey(self.context)  # create secret key
        os.makedirs(key_file_path, mode=0o775, exist_ok=True)
        # mode: A Integer value representing mode of the newly created directory. Default value Oo777 is used.
        self.sk.save(key_file_path + "/secretkey.bin")  # save secret key

        self.key_generator = heaan.KeyGenerator(self.context, self.sk)  # create public key
        self.key_generator.gen_common_keys()
        self.key_generator.save(key_file_path + "/")  # save public key

        # load secret key and public key
        # When a key is created, it can be used again to save a new key without creating a new one
        key_file_path = "keys"

        self.sk = heaan.SecretKey(self.context, key_file_path + "/secretkey.bin")  # load secret key
        self.pk = heaan.KeyPack(self.context, key_file_path + "/")  # load public key
        self.pk.load_enc_key()
        self.pk.load_mult_key()

        self.eval = heaan.HomEvaluator(self.context, self.pk)  # to load piheaan basic function
        self.dec = heaan.Decryptor(self.context)  # for decrypt
        self.enc = heaan.Encryptor(self.context)  # for encrypt

        log_slots = 15
        num_slots = 2 ** log_slots

    # Requires training once before running the service
    # Train to find and store optimal beta values
    def training(self):
        accuracy = []
        data = pd.read_csv("./Dataset.csv")

        # preprocess the dataset
        data.dropna(inplace=True)  # Remove missing values

        target = data['Loan_Status'].apply(self.trans_target_type)
        data.drop('Loan_ID', axis=1, inplace=True)  # Remove index columns

        best_accuracy = 0
        for ittr_num in range(10):  # find best beta values during iteration
            # X is the feature used to predict whether the loan will be approved, Y is the training target (whether the loan will be approved)
            x = data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
                      'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
            y = target

            # Split the dataset into training and test dataset to prevent the data leakage
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

            # Specify the categorical and numerical features respectively
            x_train_cat = x_train[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']]
            x_train_num = x_train[
                ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]

            # Use get_dummies to turned categorical features to numerical features
            x_train_dummies = pd.get_dummies(x_train_cat)

            # Drop high-correlation categorical features to improve accuracy
            x_train_dummies = x_train_dummies.drop(
                ['Gender_Male', 'Married_No', 'Education_Not Graduate', 'Self_Employed_No'], axis=1)

            # Scale numerical features and convert scaled data back to data frame format
            scaler = StandardScaler()
            scaler.fit(x_train_num)
            x_train_scaled = scaler.transform(x_train_num)
            x_train_scaled_df = pd.DataFrame(x_train_scaled)
            x_train_scaled_df.index = x_train_num.index
            x_train_scaled_df.columns = x_train_num.columns

            # Now that we're done preprocessing the categorical features and numerical categories, combine the feature columns again
            x_train_trans = pd.concat([x_train_dummies, x_train_scaled_df], axis=1)

            # Apply the same preprocessing we did to the training dataset to the test set
            x_test_cat = x_test[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']]
            x_test_num = x_test[
                ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]
            x_test_dummies = pd.get_dummies(x_test_cat)
            x_test_dummies = x_test_dummies.drop(
                ['Gender_Male', 'Married_No', 'Education_Not Graduate', 'Self_Employed_No'], axis=1)
            scaler = StandardScaler()
            scaler.fit(x_test_num)
            x_test_scaled = scaler.transform(x_test_num)
            x_test_scaled_df = pd.DataFrame(x_test_scaled)
            x_test_scaled_df.index = x_test_num.index
            x_test_scaled_df.columns = x_test_num.columns
            x_test_trans = pd.concat([x_test_dummies, x_test_scaled_df], axis=1)

            # Preprocessing is done, now we're going to train the model.
            train_n = x_train_trans.shape[0]

            X = [0] * 16
            X[0] = list(x_train_trans['Gender_Female'].values)
            X[1] = list(x_train_trans['Married_Yes'].values)
            X[2] = list(x_train_trans['Dependents_0'].values)
            X[3] = list(x_train_trans['Dependents_1'].values)
            X[4] = list(x_train_trans['Dependents_2'].values)
            X[5] = list(x_train_trans['Dependents_3+'].values)
            X[6] = list(x_train_trans['Education_Graduate'].values)
            X[7] = list(x_train_trans['Self_Employed_Yes'].values)
            X[8] = list(x_train_trans['Property_Area_Rural'].values)
            X[9] = list(x_train_trans['Property_Area_Semiurban'].values)
            X[10] = list(x_train_trans['Property_Area_Urban'].values)
            X[11] = list(x_train_trans['ApplicantIncome'].values)
            X[12] = list(x_train_trans['CoapplicantIncome'].values)
            X[13] = list(x_train_trans['LoanAmount'].values)
            X[14] = list(x_train_trans['Loan_Amount_Term'].values)
            X[15] = list(x_train_trans['Credit_History'].values)

            Y = list(y_train.values)

            msg_X = heaan.Message(self.log_slots)
            ctxt_X = heaan.Ciphertext(self.context)

            for i in range(16):
                for j in range(train_n):
                    msg_X[train_n * i + j] = X[i][j]
            self.enc.encrypt(msg_X, self.pk, ctxt_X)

            msg_Y = heaan.Message(self.log_slots)
            ctxt_Y = heaan.Ciphertext(self.context)
            for j in range(train_n):
                msg_Y[j] = Y[j]
            self.enc.encrypt(msg_Y, self.pk, ctxt_Y)

            # initial value beta
            beta = 2 * np.random.rand(17) - 1

            msg_beta = heaan.Message(self.log_slots)
            ctxt_beta = heaan.Ciphertext(self.context)

            for i in range(16):
                for j in range(train_n):
                    msg_beta[train_n * i + j] = beta[i + 1]

            for j in range(train_n):
                msg_beta[16 * train_n + j] = beta[0]

            self.enc.encrypt(msg_beta, self.pk, ctxt_beta)

            # randomly assign learning_rate
            learning_rate = 0.01
            num_steps = 100

            ctxt_next = heaan.Ciphertext(self.context)
            self.eval.add(ctxt_beta, 0, ctxt_next)
            for i in range(num_steps):
                # estimate beta_hat using function 'step' for iteration
                ctxt_next = self.step(learning_rate, ctxt_X, ctxt_Y, ctxt_next, train_n, self.log_slots, self.context,
                                      self.eval)

            # prepare test data for evaluation
            test_n = x_test.shape[0]

            X_test = [0] * 16
            X_test[0] = list(x_test_trans['Gender_Female'].values)
            X_test[1] = list(x_test_trans['Married_Yes'].values)
            X_test[2] = list(x_test_trans['Dependents_0'].values)
            X_test[3] = list(x_test_trans['Dependents_1'].values)
            X_test[4] = list(x_test_trans['Dependents_2'].values)
            X_test[5] = list(x_test_trans['Dependents_3+'].values)
            X_test[6] = list(x_test_trans['Education_Graduate'].values)
            X_test[7] = list(x_test_trans['Self_Employed_Yes'].values)
            X_test[8] = list(x_test_trans['Property_Area_Rural'].values)
            X_test[9] = list(x_test_trans['Property_Area_Semiurban'].values)
            X_test[10] = list(x_test_trans['Property_Area_Urban'].values)
            X_test[11] = list(x_test_trans['ApplicantIncome'].values)
            X_test[12] = list(x_test_trans['CoapplicantIncome'].values)
            X_test[13] = list(x_test_trans['LoanAmount'].values)
            X_test[14] = list(x_test_trans['Loan_Amount_Term'].values)
            X_test[15] = list(x_test_trans['Credit_History'].values)

            Y_test = list(y_test.values)

            msg_X_test = heaan.Message(self.log_slots)
            ctxt_X_test = heaan.Ciphertext(self.context)
            for i in range(16):
                for j in range(test_n):
                    msg_X_test[test_n * i + j] = X_test[i][j]
            self.enc.encrypt(msg_X_test, self.pk, ctxt_X_test)

            # accuracy
            ctxt_infer = self.compute_sigmoid(ctxt_X_test, ctxt_next, test_n, self.log_slots, self.eval, self.context,
                                              self.num_slots)

            res = heaan.Message(self.log_slots)
            self.dec.decrypt(ctxt_infer, self.sk, res)
            cnt = 0
            for i in range(test_n):
                if res[i].real >= 0.50:
                    if Y_test[i] == 1:
                        cnt += 1

                else:
                    if Y_test[i] == 0:
                        cnt += 1

            if best_accuracy == 0:
                best_accuracy = cnt / test_n
                ctxt_beta_best = ctxt_next

            if best_accuracy < cnt / test_n:
                best_accuracy = cnt / test_n
                ctxt_beta_best = ctxt_next
                casenum = ittr_num

            accuracy.append(cnt / test_n)

        print("Training Success!")
        self.betas = ctxt_beta_best
        return ctxt_beta_best

    def trans_target_type(self, loan_status):
        if 'Y' in loan_status:
            return 1
        else:
            return 0

    def step(self, learning_rate, ctxt_X, ctxt_Y, ctxt_beta, n, log_slots, context,
             eval):  # We customized the step function to fit our data
        '''
        ctxt_X, ctxt_Y : data for training
        ctxt_beta : initial value beta
        n : the number of row in train_data 데이터 개수수
        '''
        ctxt_rot = heaan.Ciphertext(context)
        ctxt_tmp = heaan.Ciphertext(context)

        ## step1(Update weights)
        # beta0
        ctxt_beta0 = heaan.Ciphertext(context)
        eval.left_rotate(ctxt_beta, 16 * n, ctxt_beta0)  # 16:the number of features(beta)

        # compute  ctxt_tmp = beta1*x1 + beta2*x2 + ... + beta16*x16 + beta0
        eval.mult(ctxt_beta, ctxt_X, ctxt_tmp)

        for i in range(4):
            eval.left_rotate(ctxt_tmp, n * 2 ** (3 - i), ctxt_rot)
            eval.add(ctxt_tmp, ctxt_rot, ctxt_tmp)
        eval.add(ctxt_tmp, ctxt_beta0, ctxt_tmp)

        msg_mask = heaan.Message(log_slots)
        for i in range(n):
            msg_mask[i] = 1
        eval.mult(ctxt_tmp, msg_mask, ctxt_tmp)

        ## step2
        # compute sigmoid
        approx.sigmoid(eval, ctxt_tmp, ctxt_tmp, 8.0)
        eval.bootstrap(ctxt_tmp, ctxt_tmp)
        msg_mask = heaan.Message(log_slots)
        # if sigmoid(0) -> return 0.5
        for i in range(n, self.num_slots):
            msg_mask[i] = 0.5
        eval.sub(ctxt_tmp, msg_mask, ctxt_tmp)

        ## step3
        # compute  (learning_rate/n) * (y_(j) - p_(j))
        ctxt_d = heaan.Ciphertext(context)
        eval.sub(ctxt_Y, ctxt_tmp, ctxt_d)
        eval.mult(ctxt_d, learning_rate / n, ctxt_d)

        eval.right_rotate(ctxt_d, 16 * n, ctxt_tmp)  # for beta0
        for i in range(4):
            eval.right_rotate(ctxt_d, n * 2 ** i, ctxt_rot)
            eval.add(ctxt_d, ctxt_rot, ctxt_d)
        eval.add(ctxt_d, ctxt_tmp, ctxt_d)

        ## step4
        # compute  (learning_rate/n) * (y_(j) - p_(j)) * x_(j)
        ctxt_X_j = heaan.Ciphertext(context)
        msg_X0 = heaan.Message(log_slots)
        for i in range(16 * n, 17 * n):
            msg_X0[i] = 1
        eval.add(ctxt_X, msg_X0, ctxt_X_j)
        eval.mult(ctxt_X_j, ctxt_d, ctxt_d)

        ## step5
        # compute  Sum_(all j) (learning_rate/n) * (y_(j) - p_(j)) * x_(j)
        for i in range(5):
            eval.left_rotate(ctxt_d, 2 ** (16 - i), ctxt_rot)
            eval.add(ctxt_d, ctxt_rot, ctxt_d)
        msg_mask = heaan.Message(log_slots)

        for i in range(20):
            msg_mask[i * n] = 1
        eval.mult(ctxt_d, msg_mask, ctxt_d)

        for i in range(10):
            eval.right_rotate(ctxt_d, 2 ** i, ctxt_rot)
            eval.add(ctxt_d, ctxt_rot, ctxt_d)

        ## step6
        # update beta
        eval.add(ctxt_beta, ctxt_d, ctxt_d)
        return ctxt_d

    def compute_sigmoid(self, ctxt_X, ctxt_beta, n, log_slots, eval, context, num_slots):
        '''
        ctxt_X : data for evaluation
        ctxt_beta : estimated beta from function 'step'
        n : the number of row in test_data
        '''
        ctxt_rot = heaan.Ciphertext(context)
        ctxt_tmp = heaan.Ciphertext(context)

        # beta0
        ctxt_beta0 = heaan.Ciphertext(context)
        eval.left_rotate(ctxt_beta, 16 * n, ctxt_beta0)

        # compute x * beta + beta0
        eval.mult(ctxt_beta, ctxt_X, ctxt_tmp)

        for i in range(4):
            eval.left_rotate(ctxt_tmp, n * 2 ** (3 - i), ctxt_rot)
            eval.add(ctxt_tmp, ctxt_rot, ctxt_tmp)
        eval.add(ctxt_tmp, ctxt_beta0, ctxt_tmp)

        msg_mask = heaan.Message(log_slots)
        for i in range(n):
            msg_mask[i] = 1
        eval.mult(ctxt_tmp, msg_mask, ctxt_tmp)

        # compute sigmoid
        approx.sigmoid(eval, ctxt_tmp, ctxt_tmp, 8.0)
        eval.bootstrap(ctxt_tmp, ctxt_tmp)
        msg_mask = heaan.Message(log_slots)
        for i in range(n, num_slots):
            msg_mask[i] = 0.5
        eval.sub(ctxt_tmp, msg_mask, ctxt_tmp)
        return ctxt_tmp

    def predict(self, X):  # Function that takes input (X) and computes a sigmoid to binary classify whether a loan is approved or not.
        msg_X_test = heaan.Message(self.log_slots)
        ctxt_X_test = heaan.Ciphertext(self.context)

        for i in range(16):
            msg_X_test[i] = X[i]

        self.enc.encrypt(msg_X_test, self.pk, ctxt_X_test)

        sigmoid = self.compute_sigmoid(ctxt_X_test, self.betas, 1, 15, self.eval, self.context, self.num_slots)
        res = heaan.Message(self.log_slots)
        self.dec.decrypt(sigmoid, self.sk, res)
        if res[0].real >= 0.50:
            return 1
        else:
            return 0
