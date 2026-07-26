"""
Automated Submission Packager for FlowGuard-MPC.

Author: Shyambaskar Sriram
SASTRA University - B.E. Computer Science & Engineering (AI & DS)

Description:
    Bundles technical deliverables, plots, notebook, architecture diagram, and documentation
    into a clean submission ZIP file ready for Honeywell Hackathon portal upload.
"""

import zipfile
import os

def package_submission():
    zip_filename = "FlowGuard_MPC_Honeywell_Round2_Submission_Shyambaskar_Sriram.zip"
    source_dir = r"c:\Honeywell Project\Code"
    
    files_to_include = [
        "mpc_controller.py",
        "simulator.py",
        "system_id.py",
        "run_scenarios.py",
        "app.py",
        "choke_controller_solution.ipynb",
        "README.md",
        "requirements.txt",
        "FlowGuard_MPC_Architecture_Diagram.png",
        "scenario_A_results.png",
        "scenario_B_results.png",
        "scenario_C_results.png"
    ]
    
    output_path = os.path.join(source_dir, zip_filename)
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_include:
            file_path = os.path.join(source_dir, file)
            if os.path.exists(file_path):
                zipf.write(file_path, arcname=file)
                print(f"Added to ZIP: {file}")
            else:
                print(f"Warning: File not found: {file}")
                
    print(f"\nSuccessfully created submission package: {output_path}")

if __name__ == "__main__":
    package_submission()
