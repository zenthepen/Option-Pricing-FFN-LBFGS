import json

# Load the existing notebook
with open('Double_Heston_Training_Colab.ipynb', 'r') as f:
    notebook = json.load(f)

# New cells to add
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 13. 🎯 PRICING ACCURACY EVALUATION (THE REAL TEST!)\n\n",
            "**Key Insight**: Parameter errors don't directly translate to pricing errors!\n\n",
            "Many different parameter sets can produce similar option prices. This is the \"parameter identifiability problem\" in stochastic volatility models.\n\n",
            "**What matters**: How accurately do the predicted parameters price options?"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def evaluate_pricing_accuracy(true_params, pred_params, n_samples=50):\n",
            "    \"\"\"\n",
            "    Evaluate FFN by pricing accuracy, not parameter recovery\n",
            "    \n",
            "    Returns:\n",
            "    --------\n",
            "    dict with pricing errors and visualizations\n",
            "    \"\"\"\n",
            "    print(\"=\"*70)\n",
            "    print(\"PRICING ACCURACY EVALUATION\")\n",
            "    print(\"=\"*70)\n",
            "    print(f\"\\nEvaluating {n_samples} test samples...\")\n",
            "    print(\"This measures what REALLY matters: option pricing accuracy!\\n\")\n",
            "    \n",
            "    spot = 100.0\n",
            "    r = 0.05\n",
            "    strikes = np.array([90, 95, 100, 105, 110])\n",
            "    maturities = np.array([0.25, 0.5, 1.0])\n",
            "    \n",
            "    all_pricing_errors = []\n",
            "    all_param_errors = []\n",
            "    failed_samples = 0\n",
            "    \n",
            "    for idx in range(min(n_samples, len(true_params))):\n",
            "        true_p = true_params[idx]\n",
            "        pred_p = pred_params[idx]\n",
            "        \n",
            "        sample_pricing_errors = []\n",
            "        \n",
            "        # Price all 15 options with both parameter sets\n",
            "        for K in strikes:\n",
            "            for T in maturities:\n",
            "                try:\n",
            "                    # True price\n",
            "                    model_true = DoubleHeston(\n",
            "                        S0=spot, K=K, T=T, r=r,\n",
            "                        v01=true_p[0], kappa1=true_p[1], theta1=true_p[2],\n",
            "                        sigma1=true_p[3], rho1=true_p[4],\n",
            "                        v02=true_p[5], kappa2=true_p[6], theta2=true_p[7],\n",
            "                        sigma2=true_p[8], rho2=true_p[9],\n",
            "                        lambda_j=true_p[10], mu_j=true_p[11], sigma_j=true_p[12],\n",
            "                        option_type='C'\n",
            "                    )\n",
            "                    price_true = model_true.pricing(N=64)\n",
            "                    \n",
            "                    # Predicted price\n",
            "                    model_pred = DoubleHeston(\n",
            "                        S0=spot, K=K, T=T, r=r,\n",
            "                        v01=pred_p[0], kappa1=pred_p[1], theta1=pred_p[2],\n",
            "                        sigma1=pred_p[3], rho1=pred_p[4],\n",
            "                        v02=pred_p[5], kappa2=pred_p[6], theta2=pred_p[7],\n",
            "                        sigma2=pred_p[8], rho2=pred_p[9],\n",
            "                        lambda_j=pred_p[10], mu_j=pred_p[11], sigma_j=pred_p[12],\n",
            "                        option_type='C'\n",
            "                    )\n",
            "                    price_pred = model_pred.pricing(N=64)\n",
            "                    \n",
            "                    # Percentage error\n",
            "                    if price_true > 0.01:  # Avoid division by tiny numbers\n",
            "                        pct_error = abs(price_true - price_pred) / price_true * 100\n",
            "                        sample_pricing_errors.append(pct_error)\n",
            "                \n",
            "                except Exception as e:\n",
            "                    failed_samples += 1\n",
            "                    continue\n",
            "        \n",
            "        if len(sample_pricing_errors) > 0:\n",
            "            all_pricing_errors.append(np.mean(sample_pricing_errors))\n",
            "            \n",
            "            # Parameter errors for comparison\n",
            "            param_error = np.mean(np.abs((true_p - pred_p) / (true_p + 1e-10)) * 100)\n",
            "            all_param_errors.append(param_error)\n",
            "        \n",
            "        if (idx + 1) % 10 == 0:\n",
            "            print(f\"  Progress: {idx+1}/{n_samples}\")\n",
            "    \n",
            "    results = {\n",
            "        'pricing_errors': np.array(all_pricing_errors),\n",
            "        'param_errors': np.array(all_param_errors),\n",
            "        'n_evaluated': len(all_pricing_errors),\n",
            "        'failed_samples': failed_samples\n",
            "    }\n",
            "    \n",
            "    return results\n",
            "\n",
            "print(\"✓ Pricing accuracy evaluation function defined\")"
        ]
    }
]

# Add new cells before the last cell (metadata)
notebook['cells'].extend(new_cells)

# Save updated notebook
with open('Double_Heston_Training_Colab_Enhanced.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✓ Enhanced notebook created: Double_Heston_Training_Colab_Enhanced.ipynb")
