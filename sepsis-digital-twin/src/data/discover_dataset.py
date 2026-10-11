import os
import tarfile
import shutil
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def extract_dataset():
    config = load_config()
    
    # We want to run this from the project folder or adapt path
    archive_name = config["data"]["archive_name"]
    # If path is relative to config, let's resolve it relative to workspace root (which is one level above config)
    # The config is in sepsis-digital-twin/config.yaml, and this script is in sepsis-digital-twin/src/data/discover_dataset.py
    # So workspace root is "../../" relative to this file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    archive_path = os.path.join(base_dir, archive_name)
    
    raw_dir_rel = config["paths"]["raw_dir"]
    raw_dir = os.path.join(base_dir, raw_dir_rel)
    
    set_a_target = os.path.join(raw_dir, "training_setA")
    set_b_target = os.path.join(raw_dir, "training_setB")
    
    os.makedirs(set_a_target, exist_ok=True)
    os.makedirs(set_b_target, exist_ok=True)
    
    # Check if files already extracted
    files_a = [f for f in os.listdir(set_a_target) if f.endswith('.psv')] if os.path.exists(set_a_target) else []
    files_b = [f for f in os.listdir(set_b_target) if f.endswith('.psv')] if os.path.exists(set_b_target) else []
    
    if len(files_a) == 20336 and len(files_b) == 20000:
        logger.info("Dataset already extracted: A has %d files, B has %d files.", len(files_a), len(files_b))
        return
        
    if not os.path.exists(archive_path):
        # Let's try checking in parent folder as well just in case
        alternative_path = os.path.join(os.path.dirname(base_dir), archive_name)
        if os.path.exists(alternative_path):
            archive_path = alternative_path
        else:
            raise FileNotFoundError(f"Archive not found at {archive_path} or {alternative_path}")
            
    logger.info("Extracting %s to raw directory...", archive_path)
    
    # Open and extract
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        logger.info("Total members in archive: %d", len(members))
        
        extracted_count_a = 0
        extracted_count_b = 0
        
        for member in members:
            if member.name.endswith(".psv"):
                # Extract filename
                filename = os.path.basename(member.name)
                # Determine target set
                if "training_setA" in member.name:
                    target_file = os.path.join(set_a_target, filename)
                    extracted_count_a += 1
                elif "training_setB" in member.name:
                    target_file = os.path.join(set_b_target, filename)
                    extracted_count_b += 1
                else:
                    logger.warning("Unknown folder structure for member: %s", member.name)
                    continue
                
                # Extract file contents
                f = tar.extractfile(member)
                if f is not None:
                    with open(target_file, "wb") as target_f:
                        shutil.copyfileobj(f, target_f)
                        
                if (extracted_count_a + extracted_count_b) % 5000 == 0:
                    logger.info("Extracted %d files...", extracted_count_a + extracted_count_b)
                    
        logger.info("Extraction complete. Extracted Set A: %d, Set B: %d", extracted_count_a, extracted_count_b)

if __name__ == "__main__":
    extract_dataset()
