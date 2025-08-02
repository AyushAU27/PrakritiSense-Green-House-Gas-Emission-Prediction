import pandas as pd

def preprocess_input(df):
    # Basic categorical encoding using one-hot or label encoding
    # For simplicity, we’ll do label encoding for now
    mapping_substance = {
        'carbon dioxide': 0,
        'methane': 1,
        'nitrous oxide': 2,
        'other GHGs': 3
    }

    mapping_unit = {
        'kg/2018 USD, purchaser price': 0,
        'kg CO2e/2018 USD, purchaser price': 1
    }

    mapping_source = {
        'Commodity': 0,
        'Industry': 1
    }

    df['Substance'] = df['Substance'].map(mapping_substance)
    df['Unit'] = df['Unit'].map(mapping_unit)
    df['Source'] = df['Source'].map(mapping_source)

    # Ensure column order matches what the model expects
    columns_order = [
        'Substance',
        'Unit',
        'Supply Chain Emission Factors without Margins',
        'Margins of Supply Chain Emission Factors',
        'DQ ReliabilityScore of Factors without Margins',
        'DQ TemporalCorrelation of Factors without Margins',
        'DQ GeographicalCorrelation of Factors without Margins',
        'DQ TechnologicalCorrelation of Factors without Margins',
        'DQ DataCollection of Factors without Margins',
        'Source'
    ]

    return df[columns_order]
