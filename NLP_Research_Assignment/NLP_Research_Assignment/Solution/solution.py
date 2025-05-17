import openai
import json
from tqdm import tqdm

with open("1.json", "r") as f:
    data = json.load(f)

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",  # "http://<Your api-server IP>:port"
    api_key="sk-no-key-required"
)
 
output = []

# Iterate through all patient records in the JSON file
for patient_record in tqdm(data):
    completion = client.chat.completions.create(
        model="qwen2.50-7B",
        messages=[
            {"role": "system", "content": """You are a highly specialized medical assistant designed to extract key information from clinical notes about cancer diagnoses and treatments. Your task is to analyze the provided patient EHR notes and output structured JSON data with the following details:

    Diagnosis Characteristics: Extract and organize details about the patient's cancer diagnosis, including:

    Primary cancer condition (based on these "Bladder Cancer, Breast Cancer, Colon and Rectal Cancer, Endometrial Cancer, Kidney Cancer, Leukemia, Liver Cancer, Lung Cancer, Melanoma, Non-Hodgkin Lymphoma, Pancreatic Cancer, Prostate Cancer, Thyroid Cancer, Ovarian Cancer, Esophageal Cancer, Stomach Cancer, Cervical Cancer, Head and Neck Cancer, Oral Cancer, Rectal Cancer, Hodgkin Lymphoma, Multiple Myeloma, Soft Tissue Sarcoma, Bone Cancer, Laryngeal Cancer, Vulvar Cancer, Penile Cancer, Testicular Cancer").
    Diagnosis date earliest definitive diagnosis date in MM-DD-YYYY format (this is not the date of birth). If the diagnosis date is not found.
    Histology ("Invasive ductal carcinoma, Adenocarcinoma, Squamous cell carcinoma, Non-small cell lung carcinoma (NSCLC), Clear cell renal cell carcinoma, Invasive lobular carcinoma, Basal cell carcinoma, Follicular lymphoma, Hepatocellular carcinoma, Transitional cell carcinoma, Small cell lung carcinoma, Mucinous carcinoma, Medullary thyroid carcinoma, Chondrosarcoma, Osteosarcoma, Papillary thyroid carcinoma").
    Cancer staging - [ Definition: The stage describes the extent of the cancer’s spread:
T (Tumor size/extent) – E.g., T2 means a moderate-sized tumor, T4 indicates a larger or invasive tumor.
N (Lymph node involvement) – E.g., N0 means no involvement, N1/N2 indicates progressively more nodes involved.
M (Metastasis) – E.g., M0 means no distant spread, M1 means present.
Group Stage – A single label summarizing the combination of T, N, and M, such as Stage I, Stage IIB, Stage IV, etc.
]
    Cancer-Related Medications: Extract details about medications specifically for cancer treatment, including:

    Medication name (e.g., "Paclitaxel").
    Start and end dates (in MM-DD-YYYY format; leave end date blank if ongoing).
    Intent/reason for prescribing the medication (e.g., "Neoadjuvant therapy to shrink tumor").
    Ensure the output is in the following JSON format:

    ```json
    {
      "diagnosis_characteristics": [
        {
          "primary_cancer_condition": str,
          "diagnosis_date": str,
          "histology": [str],
          "stage": {
            "T": str,
            "N": str,
            "M": str,
            "group_stage": str
          }
        }
      ],
      "cancer_related_medications": [
        {
          "medication_name": str,
          "start_date": str,
          "end_date": str,
          "intent": str
        }
      ]
    }
    ``` 

    Focus on precision and ensure all extracted details are accurate and relevant to the task. If a field is not available in the text, leave it blank or set it to null."""},
            {"role": "user", "content": f"""Given the below patient details please extract the required information and output the structured JSON data.

    {patient_record}
    """}
        ]
    )

    # Extract the JSON part from the response
    json_data = completion.choices[0].message.content.strip("```json\n").strip("```")
    try:
      # Try to parse the JSON data
      jsonx_data = json.loads(json_data)
      output.append(jsonx_data)
    except json.decoder.JSONDecodeError as e:
      print(f"Error parsing JSON for patient record: {patient_record}")
      print("Error details:", e)
      print("Response data:", json_data)
      continue  # Skip to next record on error

    # Add the key (which contains the JSON response) to the patient record for reference
    patient_record["key"] = completion.choices[0].message

# Write all the extracted JSON data to a single output file
with open("extracted_output_all_rec__patient_1.json", "w") as json_file:
    json.dump(output, json_file, indent=2)

# Optionally, print the output to verify
print(output)
