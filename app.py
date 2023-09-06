from flask import Flask, render_template, redirect, url_for, request
import pandas as pd
import joblib
import pandas as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the pre-trained model from the .pkl file
model = joblib.load('models/new_trained_rf_model.pkl')

e_s = joblib.load('models/encoder_supply.pkl')
e_c = joblib.load('models/encoder_commodity.pkl')
e_t = joblib.load('models/encoder_transport.pkl')
e_f = joblib.load('models/encoder_fuel.pkl')

df = pd.read_csv('models/Scope3_emissions_dataset.csv')
data = df.iloc[:, [1,3,5,6,10,11,12,13]]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/by_commodity')
def page1():
    return render_template('by_commodity.html')

@app.route('/by_supplier')
def page2():
    return render_template('by_supplier.html')

@app.route('/get_data_commodity/<sc>',methods=['GET']) 
def get_data_commodity(sc):
    
    filtered_rows = data[(data['Commodity_Name'] == sc)]
    result = filtered_rows.reset_index(drop=True)
    return result.to_json(orient='split', indent=4)

@app.route('/get_data_supplier/<ss>',methods=['GET']) 
def get_data_supplier(ss):
    
    filtered_rows = data[(data['Supplier_Name'] == ss)]
    result = filtered_rows.reset_index(drop=True)
    return result.to_json(orient='split', indent=4)

@app.route('/predict_by_commodity', methods=['POST'])
def predict_by_commodity():
    commodity_name = request.form['commodity_name']
    selected_supplier= request.form.get('selected_suppliers')
    Supply_Chain_Emission_Factors = float(request.form['Supply_Chain_Emission_Factors'])
    quantity = float(request.form['quantity'])
    mode_of_transport = request.form['mode_of_transport']
    fuel_type = request.form['fuel_type']
    distance_km = float(request.form['distance_km'])
    
    predicted_emission_list = []
    total_emission = 0
    result_data = []
    supplier_list = []
    
    suppliers_name = selected_supplier.split('/')
    for supplier_name in suppliers_name :
        inputs = np.array([supplier_name, commodity_name, Supply_Chain_Emission_Factors, quantity, mode_of_transport, fuel_type, distance_km])
        a = [supplier_name,commodity_name, Supply_Chain_Emission_Factors, quantity, mode_of_transport, fuel_type, distance_km]
        
        inputs=inputs.reshape(1,7)
        inputs[:,1]=e_c.transform(inputs[:,1])
        inputs[:,4]=e_t.transform(inputs[:,4])
        inputs[:,5]=e_f.transform(inputs[:,5])
        inputs[:,0]=e_s.transform(inputs[:,0])

        predicted_emission = model.predict(inputs)
        predicted_emission_list.append("{:.4f}".format(round(predicted_emission[0], 4)))
        total_emission+=predicted_emission
        supplier_list.append(supplier_name)

        result_data.append({
            'user data': a,
            'predicted CO2 emission': "{:.4f}".format(round(predicted_emission[0], 4))
        })


    filtered_rows = data[(data['Supplier_Name'].isin(supplier_list)) & (data['Commodity_Name'] == commodity_name)]


    
    return render_template('result.html',result_data=pd.DataFrame(result_data).to_html(classes='table table-striped'),total_emission=total_emission,name='commodity',edj=filtered_rows.reset_index(drop=True).to_json(orient='split', indent=4))
    

@app.route('/predict_by_supplier', methods=['POST'])
def predict_by_supplier():
    supplier_name = request.form['supplier_name']
    selected_commodity = request.form.get("selected_commodities")
    Supply_Chain_Emission_Factors = float(request.form['Supply_Chain_Emission_Factors'])
    quantity = float(request.form['quantity'])
    mode_of_transport = request.form['mode_of_transport']
    fuel_type = request.form['fuel_type']
    distance_km = float(request.form['distance_km'])
    
    predicted_emission_list = []
    total_emission = 0
    result_data = []
    commodities_list = []
    
    commodities_name = selected_commodity.split('/')
    for commodity_name in commodities_name :
        inputs = np.array([supplier_name, commodity_name, Supply_Chain_Emission_Factors, quantity, mode_of_transport, fuel_type, distance_km])
        a = [supplier_name,commodity_name, Supply_Chain_Emission_Factors, quantity, mode_of_transport, fuel_type, distance_km]
        
        inputs=inputs.reshape(1,7)
        inputs[:,1]=e_c.transform(inputs[:,1])
        inputs[:,4]=e_t.transform(inputs[:,4])
        inputs[:,5]=e_f.transform(inputs[:,5])
        inputs[:,0]=e_s.transform(inputs[:,0])

        predicted_emission = model.predict(inputs)
        predicted_emission_list.append("{:.4f}".format(round(predicted_emission[0], 4)))
        total_emission+=predicted_emission

        result_data.append({
            'user data': a,
            'predicted CO2 emission': "{:.4f}".format(round(predicted_emission[0], 4))
        })
        
        commodities_list.append(commodity_name)
    filtered_rows = data[(data['Supplier_Name'] == supplier_name) & (data['Commodity_Name'].isin(commodities_list))]

    
    return render_template('result.html',result_data=pd.DataFrame(result_data).to_html(classes='table table-striped'),total_emission=total_emission,name='commodity',edj=filtered_rows.reset_index(drop=True).to_json(orient='split', indent=4))


if __name__ == '__main__':
    app.run(debug=True)
