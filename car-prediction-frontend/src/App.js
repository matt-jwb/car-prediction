import './App.css';
import React, { useState } from 'react';

function App() {
  const [make, setMake] = useState('');
  const [model, setModel] = useState('');
  const [bodyType, setBodyType] = useState('');
  const [miles, setMiles] = useState('');
  const [age, setAge] = useState('');
  const [numOwner, setnumOwner] = useState('');
  const [predictedPrice, setPredictedPrice] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const payload = {
      "make": make,
      "model": model,
      "body_type": bodyType,
      "miles": miles,
      "age": age,
      "num_owner": numOwner
    };
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:4444/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      setPredictedPrice(data.predicted_price);
      setError('');
    } 
    catch (error) {
      console.error("Error:", error);
      setError('Failed to fetch predicted price');
      setPredictedPrice(null);
    }
    finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Car Price Predictor</h1>
      <form onSubmit={handleSubmit}>
        <input 
          type="text" 
          placeholder="Make" 
          value={make} 
          onChange={(e) => setMake(e.target.value)} 
          required 
        />
        <input 
          type="text" 
          placeholder="Model" 
          value={model} 
          onChange={(e) => setModel(e.target.value)} 
          required 
        />
        <select value={bodyType} onChange={(e) => setBodyType(e.target.value)} required>
          <option value="" disabled>Select Body Type</option>
          <option value="hatchback">Hatchback</option>
          <option value="coupe">Coupe</option>
          <option value="saloon">Saloon</option>
          <option value="suv">SUV</option>
          <option value="estate">Estate</option>
          <option value="pickup">Pickup</option>
          <option value="convertible">Convertible</option>
        </select>
        <input 
          type="number" 
          placeholder="Mileage" 
          value={miles} 
          onChange={(e) => setMiles(e.target.value)} 
          required 
        />
        <input 
          type="number" 
          placeholder="Age" 
          value={age} 
          onChange={(e) => setAge(e.target.value)} 
          required 
        />
        <input 
          type="number" 
          placeholder="Number of owners" 
          value={numOwner} 
          onChange={(e) => setnumOwner(e.target.value)} 
          required 
        />
        <button type="submit" disabled={loading}>Predict Price</button>
        </form>
        {loading && <div>Loading...</div>} {/* Loading indicator */}
        {predictedPrice && !loading && (
          <div className="result">
            <h2>Predicted Price: £{predictedPrice.toFixed(2)}</h2>
          </div>
        )}
        {error && <div className="error">{error}</div>}
      </div>
  );
}
//TODO Make sure design is responseive
//TODO Make sure error messages are handled properly and prettily
//TODO Consider input validation
//TODO Improve loading indicator

export default App;
