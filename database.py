import sqlite3
import json
import pandas as pd
from datetime import datetime
import streamlit as st
import os

# Initialize the database
def init_db():
    """Initialize the SQLite database with the necessary tables."""
    conn = sqlite3.connect('skin_analysis.db')
    c = conn.cursor()
    
    # Create analysis table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS analysis (
        analysis_id TEXT PRIMARY KEY,
        image BLOB,
        condition TEXT,
        confidence_scores TEXT,
        timestamp TEXT,
        texture_analysis TEXT,
        roi_coordinates TEXT,
        roi_prediction TEXT,
        color_analysis TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

# Save analysis to database
def save_analysis(analysis_id, image=None, condition=None, confidence_scores=None, 
                  timestamp=None, texture_analysis=None, roi_coordinates=None, 
                  roi_prediction=None, color_analysis=None, update_only=False):
    """
    Save analysis results to the database.
    
    Args:
        analysis_id (str): Unique identifier for the analysis
        image (bytes, optional): Binary image data
        condition (str, optional): Predicted skin condition
        confidence_scores (list, optional): Confidence scores for each condition
        timestamp (datetime, optional): Timestamp of analysis
        texture_analysis (dict, optional): Results of texture analysis
        roi_coordinates (tuple, optional): Coordinates of region of interest
        roi_prediction (str, optional): Prediction for the region of interest
        color_analysis (dict, optional): Results of color profile analysis
        update_only (bool): If True, update existing record instead of inserting new one
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect('skin_analysis.db')
        c = conn.cursor()
        
        # Check if record exists
        c.execute("SELECT 1 FROM analysis WHERE analysis_id = ?", (analysis_id,))
        exists = c.fetchone() is not None
        
        if exists and update_only:
            # Update existing record
            query = "UPDATE analysis SET "
            params = []
            
            # Build dynamic update query based on provided parameters
            if texture_analysis is not None:
                query += "texture_analysis = ?, "
                params.append(json.dumps(texture_analysis))
            
            if roi_coordinates is not None:
                query += "roi_coordinates = ?, "
                params.append(json.dumps(roi_coordinates))
            
            if roi_prediction is not None:
                query += "roi_prediction = ?, "
                params.append(roi_prediction)
            
            if color_analysis is not None:
                query += "color_analysis = ?, "
                params.append(json.dumps(color_analysis))
            
            # Remove trailing comma and space
            query = query.rstrip(", ")
            
            # Add WHERE clause
            query += " WHERE analysis_id = ?"
            params.append(analysis_id)
            
            # Execute update
            c.execute(query, params)
        
        elif not exists:
            # Insert new record
            c.execute('''
            INSERT INTO analysis 
            (analysis_id, image, condition, confidence_scores, timestamp, 
            texture_analysis, roi_coordinates, roi_prediction, color_analysis)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id,
                image,
                condition,
                json.dumps(confidence_scores) if confidence_scores is not None else None,
                timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp is not None else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                json.dumps(texture_analysis) if texture_analysis is not None else None,
                json.dumps(roi_coordinates) if roi_coordinates is not None else None,
                roi_prediction,
                json.dumps(color_analysis) if color_analysis is not None else None
            ))
        
        conn.commit()
        conn.close()
        return True
    
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False

# Get analysis history
def get_history():
    """
    Retrieve analysis history from the database.
    
    Returns:
        list: List of analysis records as dictionaries
    """
    try:
        conn = sqlite3.connect('skin_analysis.db')
        
        # Query the database
        query = "SELECT * FROM analysis ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        # Convert dataframe to list of dictionaries
        records = []
        for _, row in df.iterrows():
            record = {
                'analysis_id': row['analysis_id'],
                'image': row['image'],
                'condition': row['condition'],
                'confidence_scores': json.loads(row['confidence_scores']) if row['confidence_scores'] else None,
                'timestamp': row['timestamp'],
                'texture_analysis': json.loads(row['texture_analysis']) if row['texture_analysis'] else None,
                'roi_coordinates': json.loads(row['roi_coordinates']) if row['roi_coordinates'] else None,
                'roi_prediction': row['roi_prediction'],
                'color_analysis': json.loads(row['color_analysis']) if row['color_analysis'] else None
            }
            records.append(record)
        
        return records
    
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return []

# Get specific analysis by ID
def get_analysis(analysis_id):
    """
    Retrieve a specific analysis by ID.
    
    Args:
        analysis_id (str): ID of analysis to retrieve
    
    Returns:
        dict: Analysis record as dictionary or None if not found
    """
    try:
        conn = sqlite3.connect('skin_analysis.db')
        
        # Query the database
        query = "SELECT * FROM analysis WHERE analysis_id = ?"
        df = pd.read_sql_query(query, conn, params=(analysis_id,))
        
        conn.close()
        
        if df.empty:
            return None
        
        # Convert first row to dictionary
        row = df.iloc[0]
        record = {
            'analysis_id': row['analysis_id'],
            'image': row['image'],
            'condition': row['condition'],
            'confidence_scores': json.loads(row['confidence_scores']) if row['confidence_scores'] else None,
            'timestamp': row['timestamp'],
            'texture_analysis': json.loads(row['texture_analysis']) if row['texture_analysis'] else None,
            'roi_coordinates': json.loads(row['roi_coordinates']) if row['roi_coordinates'] else None,
            'roi_prediction': row['roi_prediction'],
            'color_analysis': json.loads(row['color_analysis']) if row['color_analysis'] else None
        }
        
        return record
    
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None

# Delete analysis from database
def delete_analysis(analysis_id):
    """
    Delete an analysis record from the database.
    
    Args:
        analysis_id (str): ID of analysis to delete
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect('skin_analysis.db')
        c = conn.cursor()
        
        # Delete the record
        c.execute("DELETE FROM analysis WHERE analysis_id = ?", (analysis_id,))
        
        conn.commit()
        conn.close()
        return True
    
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False
