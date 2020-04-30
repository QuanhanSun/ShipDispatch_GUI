import tkinter as ttk
import seaborn as sns
import function_design as fd
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

global ship1
global ship2


def LoadData(path, kind):
    global ship1
    global ship2

    if kind == 1:
        ship1 = pd.read_csv(path)
        ship1 = ship1.drop('timestamp', axis=1)
    else:
        ship2 = pd.read_csv(path)
        ship2 = ship2.set_index('timestamp')


def speedbefore():
    global MainWindow
    global ship1
    global ship2

    MainWindow.destroy()

    def get_comparision_chart(data1, data2, v1, v2):

        figure1 = plt.figure(figsize=(5, 3.4), dpi=100)

        select_columns = ['draft', 'engine_power', 'SPEED_KNOTS', 'Rotation', 'WindSpeed']

        if v1 == 1 and v2 == 1:
            ship1 = data1[data1['condition'] == 1]
            ship2 = data2[data2['condition'] == 1]
            ship1_clean = fd.clean(ship1)
            ship2_clean = fd.clean(ship2)
            X_1 = fd.data_transform_X(ship1_clean, select_columns)
            X_2 = fd.data_transform_X(ship2_clean, select_columns)
            Y_1 = fd.Y_value(ship1_clean)
            Y_2 = fd.Y_value(ship2_clean)
            lasso_columns_1 = fd.ship1_lasso_colums(X_1, Y_1)
            lasso_columns_2 = fd.lasso_colums_ship2(X_2, Y_2)
            coef_full_1 = fd.lasso_coef_full(X_1, Y_1, lasso_columns_1)
            coef_full_2 = fd.lasso_coef_full(X_2, Y_2, lasso_columns_2)
            emp_full_1, emr_full_1 = fd.his_engine_data(X_1)
            emp_full_2, emr_full_2 = fd.his_engine_data(X_2)
            plt.xlabel('SPEED_KNOT')
            plt.ylabel('SHIP_FUELEFFICENCY')
            plt.title('# two SHIP FULL_LOADED CONDITION')
            x = np.arange(-8, 8, 0.01)
            y1 = coef_full_1['SPEED_KNOTS^2'] * x ** 2 + coef_full_1['SPEED_KNOTS'] * x + coef_full_1[
                'engine_power SPEED_KNOTS'] * (emp_full_1) * x + coef_full_1['engine_power'] * (emp_full_1) + \
                 coef_full_1['engine_power^2'] * (emp_full_1 ** 2) + coef_full_1['Rotation'] * (emr_full_1) + \
                 coef_full_1['SPEED_KNOTS Rotation'] * (emr_full_1) * x
            y2 = coef_full_2['SPEED_KNOTS^2'] * x ** 2 + coef_full_2['SPEED_KNOTS'] * x + coef_full_2[
                'engine_power SPEED_KNOTS'] * (emp_full_2) * x + coef_full_2['engine_power'] * (emp_full_2) + \
                 coef_full_2['engine_power^2'] * (emp_full_2 ** 2) + coef_full_2['Rotation'] * (emr_full_2) + \
                 coef_full_2['SPEED_KNOTS Rotation'] * (emr_full_2) * x + coef_full_2['Rotation^2'] * (emr_full_2 ** 2)
            plt.plot(x, y1, color='r', label='ship1')
            plt.plot(x, y2, color='blue', label='ship2')
            plt.legend()

        if v1 == 0 and v2 == 0:
            ship1 = data1[data1['condition'] == 0]
            ship2 = data2[data2['condition'] == 0]
            ship1_clean = fd.clean(ship1)
            ship2_clean = fd.clean(ship2)
            X_1 = fd.data_transform_X(ship1_clean, select_columns)
            X_2 = fd.data_transform_X(ship2_clean, select_columns)
            Y_1 = fd.Y_value(ship1_clean)
            Y_2 = fd.Y_value(ship2_clean)
            lasso_columns_1 = fd.ship1_lasso_colums(X_1, Y_1)
            lasso_columns_2 = fd.lasso_colums_ship2(X_2, Y_2)
            coef_empty_1 = fd.lasso_coef_empty(X_1, Y_1, lasso_columns_1)
            coef_empty_2 = fd.lasso_coef_empty(X_2, Y_2, lasso_columns_2)
            emp_empty_1, emr_empty_1 = fd.his_engine_data(X_1)
            emp_empty_2, emr_empty_2 = fd.his_engine_data(X_2)
            plt.xlabel('SPEED_KNOT')
            plt.ylabel('SHIP_FUELEFFICENCY')
            plt.title('# two SHIP EMPTY_LOADED CONDITION')
            x = np.arange(-8, 8, 0.01)
            y1 = coef_empty_1['SPEED_KNOTS^2'] * x ** 2 + coef_empty_1['SPEED_KNOTS'] * x + coef_empty_1[
                'engine_power SPEED_KNOTS'] * (emp_empty_1) * x + coef_empty_1['engine_power'] * (emp_empty_1) + \
                 coef_empty_1['engine_power^2'] * (emp_empty_1 ** 2) + coef_empty_1['Rotation'] * (emr_empty_1)
            y2 = coef_empty_2['SPEED_KNOTS^2'] * x ** 2 + coef_empty_2['SPEED_KNOTS'] * x + coef_empty_2[
                'engine_power SPEED_KNOTS'] * (emp_empty_2) * x + coef_empty_2['engine_power'] * (emp_empty_2) + \
                 coef_empty_2['engine_power^2'] * (emp_empty_2 ** 2) + coef_empty_2['Rotation'] * (emr_empty_2) + \
                 coef_empty_2['SPEED_KNOTS Rotation'] * (emr_empty_2) * x + coef_empty_2['Rotation^2'] * (
                         emr_empty_2 ** 2)
            plt.plot(x, y1, color='r', label='ship1')
            plt.plot(x, y2, color='blue', label='ship2')
            plt.legend()

        bar1 = FigureCanvasTkAgg(figure1, SpeedBefore)
        bar1.get_tk_widget().place(x=50, y=300)

    def Label_5_change(data, v, master):
        temp = round(fd.get_ship1_speed(data, v), 2)
        Label_5 = ttk.Label(master, text=temp, font='Times 20', bg='orange')
        Label_5.place(x=240, y=190)

    def Label_6_change(data, v, master):
        temp = round(fd.get_ship2_speed(data, v), 2)
        Label_6 = ttk.Label(master, text=temp, font='Times 20', bg='orange')
        Label_6.place(x=240, y=230)

    SpeedBefore = ttk.Tk()
    SpeedBefore.title('出航前经济航速建议模块')
    SpeedBefore.geometry('600x700')

    Title = ttk.Label(SpeedBefore, text='出航前经济航速建议模块', font='Times 35')
    Title.place(x=100, y=10, height=100, width=400)

    Label_1 = ttk.Label(SpeedBefore, text='船1数据地址：', font='Times 20')
    Label_1.place(x=30, y=110, width=130)
    Ship_1 = ttk.Entry(SpeedBefore)
    Ship_1.place(x=160, y=110, width=140)
    LoadButton_1 = ttk.Button(SpeedBefore, text='载入数据', font='Times 20', command=lambda: LoadData(Ship_1.get(), 1))
    LoadButton_1.place(x=320, y=110)

    Label_2 = ttk.Label(SpeedBefore, text='船2数据地址：', font='Times 20')
    Label_2.place(x=30, y=150, width=130)
    Ship_2 = ttk.Entry(SpeedBefore)
    Ship_2.place(x=160, y=150, width=140)
    LoadButton_2 = ttk.Button(SpeedBefore, text='载入数据', font='Times 20', command=lambda: LoadData(Ship_2.get(), 2))
    LoadButton_2.place(x=320, y=150)

    v1 = ttk.IntVar()
    C1 = ttk.Radiobutton(SpeedBefore, text="空载", font='Times 20', variable=v1, value=0)
    C2 = ttk.Radiobutton(SpeedBefore, text="满载", font='Times 20', variable=v1, value=1)
    C1.place(x=420, y=110)
    C2.place(x=500, y=110)

    v2 = ttk.IntVar()
    C3 = ttk.Radiobutton(SpeedBefore, text="空载", font='Times 20', variable=v2, value=0)
    C4 = ttk.Radiobutton(SpeedBefore, text="满载", font='Times 20', variable=v2, value=1)
    C3.place(x=420, y=150)
    C4.place(x=500, y=150)

    Label_3 = ttk.Label(SpeedBefore, text='船1建议经济航速：', font='Times 20')
    Label_3.place(x=25, y=190)
    Label_4 = ttk.Label(SpeedBefore, text='船2建议经济航速：', font='Times 20')
    Label_4.place(x=25, y=230)

    CacuButton_1 = ttk.Button(SpeedBefore, text='输出', font='Times 20',
                              command=lambda: Label_5_change(ship1, v1.get(), SpeedBefore))
    CacuButton_2 = ttk.Button(SpeedBefore, text='输出', font='Times 20',
                              command=lambda: Label_6_change(ship2, v2.get(), SpeedBefore))
    CacuButton_1.place(x=360, y=190)
    CacuButton_2.place(x=360, y=230)

    DrawButton_1 = ttk.Button(SpeedBefore, text='绘图', font='Times 20',
                              command=lambda: get_comparision_chart(ship1, ship2, v1.get(), v2.get()))
    DrawButton_1.place(x=420, y=190)

    SpeedBefore.mainloop()


def speedduring():
    global MainWindow

    MainWindow.destroy()

    def create_charts():

        x1 = float(Power.get())
        x2 = float(Rotating.get())
        x3 = int(v2.get())
        x4 = int(v1.get())

        figure1 = plt.figure(figsize=(5, 3.4), dpi=100)

        if x3 == 1 and x4 == 1:
            x1 = (x1 - 9573.89) / 1363.21
            x2 = (x2 - 44.52) / 1.95

            plt.xlabel('SPEED_KNOT')
            plt.ylabel('SHIP_FUELEFFICENCY')
            plt.title('#1 SHIP FULL_LOADED CONDITION')
            x = np.arange(-6, 6, 0.01)
            y = 0.022 * x ** 2 - 0.191 * x - 0.017 * (x1) * x + 0.181 * (x1) + 0.006 * (x1 ** 2) + 0.028 * (
                x2) - 0.005 * (x2) * x
            plt.plot(x, y, color='blue', label='speed vs ship_fuelefficency')

        if x3 == 1 and x4 == 0:
            x1 = (x1 - 7993) / 2579.95
            x2 = (x2 - 43.88) / 4.92
            plt.xlabel('SPEED_KNOT')
            plt.ylabel('SHIP_FUELEFFICENCY')
            plt.title('#1 SHIP EMPTY_LOADED CONDITION')
            x = np.arange(-6, 6, 0.01)
            y = 0.027 * x ** 2 - 0.167 * x - 0.042 * (x1) * x + 0.008 * (x1 ** 2) + 0.025 * (x2) + 0.087 * (x2)
            plt.plot(x, y, color='blue', label='speed vs ship_fuelefficency')

        if x3 == 2 and x4 == 1:
            x1 = (x1 - 9798.23) / 2344.71
            x2 = (x2 - 44.63) / 3.51
            plt.xlabel('SPEED_KNOT')
            plt.ylabel('SHIP_FUELEFFICENCY')
            plt.title('#2 SHIP FULL_LOADED CONDITION')
            x = np.arange(-8, 8, 0.01)
            y = 0.024 * x ** 2 - 0.226 * x + 0.012 * (x1) * x + 0.145 * (x1) + 0.007 * (x1 ** 2) + 0.028 * (
                x2) + 0.065 * (
                    x2) - 0.015 * ((x2) ** 2)
            plt.plot(x, y, color='blue', label='speed vs ship_fuelefficency')

        if x3 == 2 and x4 == 0:
            x1 = (x1 - 8638.723) / 2035.38
            x2 = (x2 - 43.44) / 3.39
            plt.xlabel('SPEED_KNOT')
            plt.ylabel('SHIP_FUELEFFICENCY')
            plt.title('#2 SHIP EMPTY_LOADED CONDITION')
            x = np.arange(-10, 10, 0.01)
            y = 0.007 * x ** 2 - 0.122 * x - 0.026 * (x1) * x + 0.166 * (x1) + 0.015 * (x1 ** 2) + 0.028 * (
                x2) * x + 0.065 * (x2) - 0.015 * ((x2) ** 2)
            plt.plot(x, y, color='blue', label='speed vs ship_fuelefficency')

        bar1 = FigureCanvasTkAgg(figure1, SpeedDuring)
        bar1.get_tk_widget().place(x=50, y=240)

    def show_speed(Label_4):
        shipoptspeed = round(fd.get_speed(float(Power.get()), float(Rotating.get()), v2.get(), v1.get()), 2)
        Label_4.destroy()
        Label_4 = ttk.Label(SpeedDuring, text=shipoptspeed, font='Times 20')
        Label_4.place(x=250, y=190)

    SpeedDuring = ttk.Tk()
    SpeedDuring.title('在航中经济航速建议模块')
    SpeedDuring.geometry('600x600')

    Title = ttk.Label(SpeedDuring, text='在航中经济航速建议模块', font='Times 35')
    Title.place(x=100, y=10, height=100, width=400)

    Label_1 = ttk.Label(SpeedDuring, text='主机功率：', font='Times 20')
    Label_1.place(x=30, y=110, width=100)
    Power = ttk.Entry(SpeedDuring)
    Power.place(x=130, y=110)

    Label_2 = ttk.Label(SpeedDuring, text='主机转速：', font='Times 20')
    Label_2.place(x=30, y=150, width=100)
    Rotating = ttk.Entry(SpeedDuring)
    Rotating.place(x=130, y=150)

    v1 = ttk.IntVar()
    C1 = ttk.Radiobutton(SpeedDuring, text="空载", font='Times 20', variable=v1, value=0)
    C2 = ttk.Radiobutton(SpeedDuring, text="满载", font='Times 20', variable=v1, value=1)
    C1.place(x=390, y=110)
    C2.place(x=460, y=110)

    v2 = ttk.IntVar()
    C3 = ttk.Radiobutton(SpeedDuring, text="船1", font='Times 20', variable=v2, value=1)
    C4 = ttk.Radiobutton(SpeedDuring, text="船2", font='Times 20', variable=v2, value=2)
    C3.place(x=390, y=150)
    C4.place(x=460, y=150)

    Label_3 = ttk.Label(SpeedDuring, text='所选船只建议经济航速：', font='Times 20')
    Label_3.place(x=25, y=190)
    Label_4 = ttk.Label(SpeedDuring, text="", font='Times 20')
    Label_4.place(x=250, y=190)

    CacuButton_1 = ttk.Button(SpeedDuring, text='输出', font='Times 20', command=lambda: show_speed(Label_4))
    CacuButton_1.place(x=350, y=190)
    DrawButton_1 = ttk.Button(SpeedDuring, text='绘图', font='Times 20', command=lambda: create_charts())
    DrawButton_1.place(x=400, y=190)

    SpeedDuring.mainloop()


def comparedata():
    global MainWindow

    MainWindow.destroy()

    def create_pic(data1, data2, v, type):
        figure1 = plt.figure(figsize=(5, 3.3), dpi=100)
        data = fd.new_data(data1, data2)
        data = data[data['condition'] == v]
        sns.boxplot(x="condition", y=type, hue="name", data=data)
        plt.autoscale(axis='y')
        bar1 = FigureCanvasTkAgg(figure1, CompareData)
        bar1.get_tk_widget().place(x=50, y=220, width=500, height=370)

    # def get_fuel_consumption(data1, data2, v1):
    #     figure1 = plt.figure(figsize=(5, 3.6), dpi=100)
    #     data = fd.new_data(data1, data2)
    #     data = data[data['condition'] == v1]
    #     sns.boxplot(x="condition", y="fuel_consumption", hue="name", data=data)
    #     bar1 = FigureCanvasTkAgg(figure1, CompareData)
    #     bar1.get_tk_widget().place(x=50, y=220)

    # def get_fuel_efficiency(data1, data2, v1):
    #     figure1 = plt.figure(figsize=(5, 3.5), dpi=100)
    #     data = fd.new_data(data1, data2)
    #     data = data[data['condition'] == v1]
    #     plt.ylim((0, 400))
    #     sns.boxplot(x="condition", y="ship_FuelEfficiency", hue="name", data=data)
    #     bar1 = FigureCanvasTkAgg(figure1, CompareData)
    #     bar1.get_tk_widget().place(x=50, y=220)
    #
    # def get_draft(data1, data2, v1):
    #     figure1 = plt.figure(figsize=(5, 3.6), dpi=100)
    #     data = fd.new_data(data1, data2)
    #     data = data[data['condition'] == v1]
    #     sns.boxplot(x="condition", y="draft", hue="name", data=data)
    #     bar1 = FigureCanvasTkAgg(figure1, CompareData)
    #     bar1.get_tk_widget().place(x=50, y=220)
    #
    # def get_speed(data1, data2, v1):
    #     figure1 = plt.figure(figsize=(5, 3.6), dpi=100)
    #     data = fd.new_data(data1, data2)
    #     data = data[data['condition'] == v1]
    #     sns.boxplot(x="condition", y="SPEED_KNOTS", hue="name", data=data)
    #     bar1 = FigureCanvasTkAgg(figure1, CompareData)
    #     bar1.get_tk_widget().place(x=50, y=220)
    #
    # def get_power(data1, data2, v1):
    #     figure1 = plt.figure(figsize=(5, 3.6), dpi=100)
    #     data = fd.new_data(data1, data2)
    #     data = data[data['condition'] == v1]
    #     sns.boxplot(x="condition", y="engine_power", hue="name", data=data)
    #     bar1 = FigureCanvasTkAgg(figure1, CompareData)
    #     bar1.get_tk_widget().place(x=50, y=220)

    CompareData = ttk.Tk()
    CompareData.title('多船数据比较分析')
    CompareData.geometry('600x600')

    Title = ttk.Label(CompareData, text='多船数据比较分析', font='Times 35')
    Title.place(x=100, y=10, height=100, width=400)

    Label_1 = ttk.Label(CompareData, text='船1数据地址：', font='Times 20')
    Label_1.place(x=30, y=110, width=130)
    Ship_1 = ttk.Entry(CompareData)
    Ship_1.place(x=160, y=110, width=140)
    LoadButton_1 = ttk.Button(CompareData, text='载入数据', font='Times 20', command=lambda: LoadData(Ship_1.get(), 1))
    LoadButton_1.place(x=320, y=110)

    Label_2 = ttk.Label(CompareData, text='船2数据地址：', font='Times 20')
    Label_2.place(x=30, y=150, width=130)
    Ship_2 = ttk.Entry(CompareData)
    Ship_2.place(x=160, y=150, width=140)
    LoadButton_2 = ttk.Button(CompareData, text='载入数据', font='Times 20', command=lambda: LoadData(Ship_2.get(), 2))
    LoadButton_2.place(x=320, y=150)

    v = ttk.IntVar()
    C1 = ttk.Radiobutton(CompareData, text="空载", font='Times 20', variable=v, value=0)
    C2 = ttk.Radiobutton(CompareData, text="满载", font='Times 20', variable=v, value=1)
    C1.place(x=460, y=110)
    C2.place(x=460, y=150)

    Button_1 = ttk.Button(CompareData, text='燃油消耗', font='Times 20',
                          command=lambda: create_pic(ship1, ship2, v.get(), type='fuel_consumption'))
    Button_1.place(x=75, y=190)
    Button_2 = ttk.Button(CompareData, text='吨/海里消耗', font='Times 20',
                          command=lambda: create_pic(ship1, ship2, v.get(), type='ship_FuelEfficiency'))
    Button_2.place(x=175, y=190)
    Button_3 = ttk.Button(CompareData, text='吃水', font='Times 20', command=lambda: create_pic(ship1, ship2, v.get(),
                          type='draft'))
    Button_3.place(x=300, y=190)
    Button_4 = ttk.Button(CompareData, text='航速', font='Times 20', command=lambda: create_pic(ship1, ship2, v.get(),
                          type='SPEED_KNOTS'))
    Button_4.place(x=360, y=190)
    Button_5 = ttk.Button(CompareData, text='主机功率', font='Times 20', command=lambda: create_pic(ship1, ship2, v.get(),
                          type='engine_power'))
    Button_5.place(x=420, y=190)

    CompareData.mainloop()


MainWindow = ttk.Tk()
MainWindow.title('船队决策优化模块')
MainWindow.geometry('600x600')

Title = ttk.Label(MainWindow, text='船队决策优化模块', font='Times 35')
Title.place(x=100, y=10, height=100, width=400)

CompareButton = ttk.Button(MainWindow, text='多船数据比较分析', font='Times 25', command=lambda: comparedata())
CompareButton.place(x=150, y=150, height=100, width=300)

SpeedBeforeButton = ttk.Button(MainWindow, text='出航前经济航速建议模块', font='Times 25', command=lambda: speedbefore())
SpeedBeforeButton.place(x=150, y=300, height=100, width=300)

SpeedDuringButton = ttk.Button(MainWindow, text='在航中经济航速建议模块', font='Times 25', command=lambda: speedduring())
SpeedDuringButton.place(x=150, y=450, height=100, width=300)

MainWindow.mainloop()
