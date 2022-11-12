"""
Pandas y Numpy 
nos ayuda a leer los archivos .csv 
y asigana los datos de estos archivos en una 
estructura de datos que podamos manipular 
"""
import pandas as pd 
import numpy as np
"""
matplotlib
la usamos para generar las graficas necesarias 
para el analisis de los datos 
"""
import matplotlib.pyplot as plt
"""
sklearn
la usamos parea generar y entrenar modelos 
de regrecion lienal para entender el comportamiendo de 
los datos y poder generar predicciones 
"""
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# leemos y guaradamos los datos de los archivos .csv en Dataframes
data = pd.read_csv("data.csv", sep=";")
pib_per_regiones = pd.read_csv("pib-per-regiones.csv", sep=";")
pib_regiones = pd.read_csv("pib regiones.csv", sep=";")

regiones=["Caribe","Oriental","Central","Pacífica","Bogotá","Antioquia","Valle del Cauca"]

"""
con esta funcion buacamos predecir a partir de los datos obtenidos en los dataframes 
el comportamiento del PIB en cada region a mediano plazon 
"""
def predict_pib_region():
    # solicitamos y validamos la region 
    print(" predicon del PIB percapita de una region")
    print(" Regiones")
    i=1
    for r in regiones:
        print("#",i,":",r)
        i+=1
    indr=int(input("\n elija la region a predecir: "))

    reg=regiones[indr-1]

    # ontenemos las columnas correspondientes de la tabla 
    x=pib_regiones["Anios"].values.reshape(-1, 1)
    y1=pib_regiones[reg].values.reshape(-1, 1)
    y2=pib_per_regiones[reg].values.reshape(-1, 1)

    # estandarizamos los datos ibtenidos del Dataframe
    sc_x = StandardScaler()
    sc_y1 = StandardScaler()
    sc_y2 = StandardScaler()

    X_std = sc_x.fit_transform(x)
    y1_std = sc_y1.fit_transform(y1)
    y2_std = sc_y2.fit_transform(y2)

    # declaramos las regrecionde de :
    slr_pib_regiones = LinearRegression() # PIB por regiones
    # entrenamos el modelo de regrecion slr_pib_regiones 
    slr_pib_regiones.fit(X_std, y1_std) 

    slr_pib_per_regiones = LinearRegression() # PIB per-capita regiones
    # entrenamos el modelo de regrecion slr_pib_per_regiones 
    slr_pib_per_regiones.fit(X_std, y2_std) 

    # solicitamos el año a predecir 
    anio_predict = int(input("ingresa el año a predecir: "))

    # estandarizamos el año a predecir 
    anio_predict_std = sc_x.transform(np.array([anio_predict]).reshape(-1,1))
    
    # realizamos la predicciones 
    pib_predict=sc_y1.inverse_transform(slr_pib_regiones.predict(anio_predict_std))
    pib_per_predict=sc_y2.inverse_transform(slr_pib_per_regiones.predict(anio_predict_std))
    
    # mostramos resultados 
    print("\n prediciones PIB en la region ",regiones[indr-1]," en ele anio ",anio_predict)
    print("{:^20}{:^20,}{:^20}".format("TOTAL",round(pib_predict[0][0],2),"Miles de millones de pesos"))
    print("{:^20}{:^20,}{:^5}".format("Per-Capita",round(pib_per_predict[0][0],2),"Pesos"))

    ingr_mes = pib_per_predict[0][0]/12
    print("\n Se deduce que los ingresos mensuales por persona seran aproximandamente de ")
    print("{:^20,}{:^5}".format(round(ingr_mes,2),"Pesos"))


"""
esta funcion obtiene el modelo de regrecion entre el PIB y el indice de 
pobreza multidimencional para entendir como uno influye en el otro 
"""
def predict_pm_pib_per():
    print(" prediccione de indices de pobreza a mediano plazo ")
    print(" Regiones ")
    i=1
    for r in regiones:
        print("#",i,":",r)
        i+=1
    indr=int(input("\n elija la region a evaluar: "))

    pb_reg="PB-MLT-"+regiones[indr-1]
    pib_reg="PIB-"+regiones[indr-1]

    # ontenemos las columnas correspondientes de la tabla 
    anios=data["Anios"].values.reshape(-1, 1)

    PIB_region = data[pib_reg].values.reshape(-1, 1)
    PB_MLT_region = data[pb_reg].values.reshape(-1, 1)
    

    # estandarizamos los datos ibtenidos del Dataframe
    sc_anios = StandardScaler()
    sc_PIB_region = StandardScaler()
    sc_PB_MLT_region = StandardScaler()
    
    PIB_region_std = sc_PIB_region.fit_transform(PIB_region)
    PB_MLT_region_std = sc_PB_MLT_region.fit_transform(PB_MLT_region)

    # declaramos las regrecionde de :
    slr_PB_MLT = LinearRegression()  # pobresa multidimencional
    slr_PB_MLT.fit(PIB_region_std, PB_MLT_region_std ) # entrenamos el modelo de regrecion

    plt.scatter(PIB_region_std,PB_MLT_region_std)
    plt.plot(PIB_region_std,slr_PB_MLT.predict(PB_MLT_region_std), color='red')
    plt.ylabel("PB-MLT")
    plt.xlabel("PIB-region")
    plt.show()
    
#predict_pm_pib_per()
#predict_pib_region()