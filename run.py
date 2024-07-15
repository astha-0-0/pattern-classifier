#importing all necessary libraries
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import simpledialog, filedialog
import tkinter.messagebox
import pickle
import os.path
import PIL
import PIL.Image, PIL.ImageDraw
import cv2 as cv
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

#importing important models from sklearn
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class Classifier:
    def __init__(self):
        #a new list to store class name
        self.class_names = []

        #a dictionary to store counts of images in a class
        self.class_counters = {}
        self.clf = None #act as a classifier model
        self.projectname = None
        self.root = None
        self.image1 = None

        self.status_label = None
        self.canvas = None
        self.draw = None

        self.brush_width = 15

        # Calling methods to set up the project
        self.classes_prompt()
        self.init_gui()

    def classes_prompt(self):
        # Prompting for project name and number of classes
        msg = Tk()
        msg.withdraw()

        self.projectname = simpledialog.askstring("Project Name", "Enter project name!", parent=msg)
        num_classes = simpledialog.askinteger("Number of Classes", "How many classes do you want to classify?", parent=msg)
        
        #to load already present file if available else creates a new project
        if os.path.exists(self.projectname):
            with open(f"{self.projectname}/{self.projectname}_data.pickle", "rb") as f:
                data = pickle.load(f)
            self.class_names = data['class_names']
            self.class_counters = data['class_counters']
            self.clf = data['clf']
            self.projectname = data['pname']
        else:
            #creating new directory for named class
            for i in range(num_classes):
                class_name = simpledialog.askstring(f"Class {i+1}", f"Enter name of class {i+1}?", parent=msg)
                self.class_names.append(class_name)
                self.class_counters[class_name] = 1

            self.clf = LinearSVC() #initial model loading

            #creating project and class directory
            os.mkdir(self.projectname)
            os.chdir(self.projectname)
            for class_name in self.class_names:
                os.mkdir(class_name)
            os.chdir("..")


    #tkinter GUI method
    def init_gui(self):
        WIDTH = 500
        HEIGHT = 500
        WHITE = (255, 255, 255)

        self.root = Tk()
        self.root.title(f"Classifier - {self.projectname}") #this is for root window

        #creating frontend for pattern insertion
        self.canvas = Canvas(self.root, width=WIDTH-10, height=HEIGHT-10, bg="#AAC9CE")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        #creating PIL image object for drawing
        self.image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        #for buttons in GUUI
        btn_frame = Frame(self.root)
        btn_frame.pack(fill=X, side=BOTTOM)

        for i, class_name in enumerate(self.class_names):
            btn = Button(btn_frame, text=class_name, command=lambda name=class_name: self.save(name))
            btn.grid(row=0, column=i, sticky=W + E)

        #for brush manipulation and control
        bm_btn = Button(btn_frame, text="Brush-", command=self.brushminus)
        bm_btn.grid(row=1, column=0, sticky=W + E)

        clear_btn = Button(btn_frame, text="Clear", command=self.clear)
        clear_btn.grid(row=1, column=1, sticky=W + E)

        bp_btn = Button(btn_frame, text="Brush+", command=self.brushplus)
        bp_btn.grid(row=1, column=2, sticky=W + E)

        train_btn = Button(btn_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=2, column=0, sticky=W + E)

        save_btn = Button(btn_frame, text="Save Model", command=self.save_model)
        save_btn.grid(row=2, column=1, sticky=W + E)

        load_btn = Button(btn_frame, text="Load Model", command=self.load_model)
        load_btn.grid(row=2, column=2, sticky=W + E)

        # change_btn = Button(btn_frame, text="Change Model", command=self.rotate_model)
        # change_btn.grid(row=3, column=0, sticky=W + E)

        predict_btn = Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=3, column=0, sticky=W + E)

        save_everything_btn = Button(btn_frame, text="Save Everything", command=self.save_everything)
        save_everything_btn.grid(row=3, column=1, sticky=W + E)

        self.status_label = Label(btn_frame, text=f"Current Model: {type(self.clf).__name__}")
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=4, column=1, sticky=W + E)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    #for drawing on canvas (user end)
    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)
        self.draw.rectangle([x1, y2, x2 + self.brush_width, y2 + self.brush_width], fill="black", width=self.brush_width)

    #saves drawn images to class folder created
    def save(self, class_name):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.ANTIALIAS)
        img.save(f"{self.projectname}/{class_name}/{self.class_counters[class_name]}.png", "PNG")
        self.class_counters[class_name] += 1
        self.clear()

    def brushminus(self):
        if self.brush_width > 1:
            self.brush_width -= 1

    def brushplus(self):
        self.brush_width += 1

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white")

    
        
    #training & testing of the models
    def train_model(self):
        img_list = []
        class_list = []

        for class_name in self.class_names:
            for x in range(1, self.class_counters[class_name]):
                img = cv.imread(f"{self.projectname}/{class_name}/{x}.png")[:, :, 0]
                img = img.reshape(2500)
                img_list.append(img)
                class_list.append(self.class_names.index(class_name))

        img_list = np.array(img_list)
        class_list = np.array(class_list)

        self.clf.fit(img_list, class_list)

        tkinter.messagebox.showinfo("Classifier", "Model trained successfully!", parent=self.root)

        # Split data into training and testing sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(img_list, class_list, test_size=0.2, random_state=42)

        accuracies = []
        f1_scores = []
        recalls = []
        precisions = []
        models = ['LinearSVC','GaussianNB','DecisionTreeClassifier', 'KNeighborsClassifier','RandomForestClassifier','LogisticRegression']
        models_list = []  # To store trained models
        all_predictions=[]

        for model_name in models:
            model = self.get_model_instance(model_name)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = metrics.f1_score(y_test, y_pred, average='macro')
            recall = metrics.recall_score(y_test, y_pred, average='macro')
            precision = metrics.precision_score(y_test, y_pred, average='macro')
            accuracy = model.score(X_test, y_test)
            f1_scores.append(f1)
            recalls.append(recall)
            precisions.append(precision)
            accuracies.append(accuracy)
            models_list.append(model)
            all_predictions.append(y_pred)

            # Calculate ROC curve
            y_pred_proba = model.predict_proba(X_test)
            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = metrics.auc(fpr, tpr)
            
            # Plot ROC curve
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.show()

            # Calculate Specificity
            # tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
            # specificity = tn / (tn + fp)
            # print(f'Specificity for {model_name}: {specificity}')

            
        
       

        # Plotting the accuracy graph
        plt.figure(figsize=(10, 6))
        plt.bar(models, accuracies, color='skyblue')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of Different Models')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plotting the F1-score graph
        plt.figure(figsize=(10, 6))
        plt.bar(models, f1_scores, color='orange')
        plt.xlabel('Model')
        plt.ylabel('F1 Score')
        plt.title('F1 Score of Models')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plotting the recall graph
        plt.figure(figsize=(10, 6))
        plt.bar(models, recalls, color='green')
        plt.xlabel('Model')
        plt.ylabel('Recall')
        plt.title('Recall of Models')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plotting the precision graph
        plt.figure(figsize=(10, 6))
        plt.bar(models, precisions, color='red')
        plt.xlabel('Model')
        plt.ylabel('Precision')
        plt.title('Precision of Models')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        

    def get_model_instance(self, model_name):
        if model_name == 'LinearSVC':
            base_model = LinearSVC()
            calibrated_model = CalibratedClassifierCV(base_model)  # Wrap LinearSVC with CalibratedClassifierCV
            return calibrated_model
        elif model_name == 'GaussianNB':
            return GaussianNB()
        elif model_name == 'DecisionTreeClassifier':
            return DecisionTreeClassifier()
        elif model_name == 'KNeighborsClassifier':
            return KNeighborsClassifier()
        elif model_name == 'RandomForestClassifier':
            return RandomForestClassifier()
        elif model_name == 'LogisticRegression':
            return LogisticRegression()

    def predict(self):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.ANTIALIAS)
        img.save("predictshape.png", "PNG")

        img = cv.imread("predictshape.png")[:, :, 0]
        img = img.reshape(2500)
        prediction = self.clf.predict([img])
        class_name = self.class_names[prediction[0]]
        tkinter.messagebox.showinfo("Classifier", f"The drawing is probably a {class_name}", parent=self.root)

    def rotate_model(self):
        if isinstance(self.clf, LinearSVC):
            self.clf = KNeighborsClassifier()
        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf= RandomForestClassifier()
        elif isinstance(self.clf, RandomForestClassifier):
            self.clf= LinearSVC()
        

        self.status_label.config(text=f"Current Model: {type(self.clf).__name__}")

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension="pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.clf, f)
        tkinter.messagebox.showinfo("Classifier", "Model saved!", parent=self.root)

    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, "rb") as f:
            self.clf = pickle.load(f)
        tkinter.messagebox.showinfo("Classifier", "Model loaded!", parent=self.root)

    def save_everything(self):
        data = {"class_names": self.class_names, "class_counters": self.class_counters,
                "clf": self.clf, "pname": self.projectname}
        with open(f"{self.projectname}/{self.projectname}_data.pickle", "wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo("Classifier", "Project successfully saved!", parent=self.root)

    def on_closing(self):
        answer = tkinter.messagebox.askyesnocancel("Quit?", "Do you want to save this project?", parent=self.root)
        if answer is not None:
            if answer:
                self.save_everything()
            self.root.destroy()
            exit()