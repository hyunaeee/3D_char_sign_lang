import os
import json
import glob
from pathlib import Path

def extract_morpheme_data(json_file_path):
    
    """
    Extract time periods and coordinates for each sign language morpheme from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        dict: Dictionary containing extracted morpheme data
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract basic metadata
        result = {
            "file_id": data["metadata"]["id"],
            "korean_text": data.get("korean_text", ""),
            "morphemes": []
        }
        
        # Check if sign_script and sign_gestures_both exist
        if "sign_script" in data and "sign_gestures_both" in data["sign_script"]:
            for gesture in data["sign_script"]["sign_gestures_both"]:
                if "gloss_id" in gesture:
                    morpheme_info = {
                        "gloss_id": gesture["gloss_id"],
                        "start_frame": gesture.get("start_frame"),
                        "end_frame": gesture.get("end_frame")
                    }
                    
                    # Extract coordinates if available
                    if "coordinates" in gesture:
                        morpheme_info["coordinates"] = gesture["coordinates"]
                    
                    result["morphemes"].append(morpheme_info)
        
        return result
    
    except Exception as e:
        print(f"Error processing {json_file_path}: {str(e)}")
        return None

def process_json_files(base_dir):
    """
    Process all JSON files in the given directory and its subdirectories.
    
    Args:
        base_dir (str): Base directory containing JSON files
        
    Returns:
        list: List of dictionaries containing extracted morpheme data
    """
    all_results = []
    
    # Use glob to find all JSON files recursively
    json_files = glob.glob(os.path.join(base_dir, "**/*.json"), recursive=True)
    
    for json_file in json_files:
        result = extract_morpheme_data(json_file)
        if result:
            all_results.append(result)
            
            # Print basic information for each file
            print(f"Processed: {result['file_id']}")
            print(f"Korean text: {result['korean_text']}")
            print(f"Number of morphemes: {len(result['morphemes'])}")
            
            # Print details for each morpheme
            for i, morpheme in enumerate(result['morphemes']):
                print(f"  Morpheme {i+1}: {morpheme['gloss_id']}")
                if morpheme.get('start_frame') is not None and morpheme.get('end_frame') is not None:
                    print(f"    Time period: frames {morpheme['start_frame']} to {morpheme['end_frame']}")
                if "coordinates" in morpheme:
                    print(f"    Coordinates available: Yes")
                print()
            
            print("-" * 80)
    
    return all_results

def main():
    # Base directory containing the JSON files
    # You may need to adjust this path based on your actual directory structure
    base_dir = "114.재난 안전 정보 전달을 위한 수어영상 데이터/01.데이터/2.Validation/라벨링데이터/03_JSON_VL"
    
    # Check if the directory exists
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        print("Please provide the correct path to the JSON files.")
        return
    
    # Process all JSON files
    results = process_json_files(base_dir)
    
    print(f"Total files processed: {len(results)}")

if __name__ == "__main__":
    main()
