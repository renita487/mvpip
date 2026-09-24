# 🚢 Quantum-Inspired Fuel Consumption Prediction and Green Fleet Optimization

**Smart India Hackathon (SIH) MVP Web Application**

A cutting-edge decision support system for maritime decarbonization, integrating **naval hydrodynamics**, **machine learning surrogate models**, and **quantum-inspired optimization** to predict vessel fuel consumption, quantify greenhouse gas (GHG) emissions, and optimize fleet-wide speed scheduling.

---

## 🌟 Key Features

1. **⚡ Page 1: Precision Fuel Consumption Prediction**
   - Physics-based **Admiralty power formulation** ($P \propto \Delta^{2/3} v^3$) adjusted for cargo displacement, engine SFOC, and weather sea states.
   - Interactive vessel profiles: Container Ship (8,000 TEU), Capesize Bulk Carrier, VLCC Tanker, and General Cargo.
   - Dynamic **Speed-Fuel cubic sensitivity curve** & instant comparison with Slow Steaming (-2 knots).

2. **🌿 Page 2: IMO Emission & Cost Analysis**
   - IMO GHG compliance with standard emission factor ($3.17 \text{ t CO}_2 / \text{t Fuel}$).
   - Direct fuel cost calculation with live customizable bunker price (default $\$600/\text{tonne}$).
   - **Carbon Intensity Indicator (CII)** grading simulator (Rating A to E) and EU ETS Carbon Tax impact.
   - Side-by-side Normal vs Slow Steaming savings audit.

3. **⚛️ Page 3: Basic Green Fleet Optimization Demo**
   - Multi-vessel cargo demand distribution (1 to 5 ships).
   - Speed options comparison matrix (**12, 14, 16, 18 knots**) highlighting the optimal green strategy.
   - **Quantum-Inspired Simulated Annealing (QISA)** heuristic engine discovering global energy minima.
   - Pareto trade-off curve (Voyage Duration vs Total Fleet Cost & Emissions).

4. **📊 Page 4: ML Telemetry & Transparency**
   - 1,200 simulated voyage telemetry logs with sensor noise.
   - Trained Scikit-Learn Random Forest Regressor ($R^2 > 98\%$).
   - Feature importance rankings demonstrating non-linear speed dominance.

---

## 🛠️ Technology Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Visual Analytics**: [Plotly](https://plotly.com/) (with responsive dark theme)
- **Scientific Computing**: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **Surrogate AI/ML**: [Scikit-Learn](https://scikit-learn.org/)
- **Domain Modeling**: IMO 2023 GHG Strategy & Admiralty Naval Architecture

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
py -m pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
py -m streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.
