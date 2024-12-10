import os

def copy_code_to_txt(repo_path, output_file, file_extensions=None):
    if file_extensions is None:
        file_extensions = ['.py']  # Default to Python files only

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(f'File: {file_path}\n')
                            outfile.write(infile.read())
                            outfile.write('\n' + '-' * 80 + '\n')  # Separator between files
                    except FileNotFoundError:
                        print(f"File not found: {file_path}")
                    except PermissionError:
                        print(f"Permission denied: {file_path}")
                    except UnicodeDecodeError:
                        print(f"Unable to decode {file_path} due to encoding issues")
                    except Exception as e:
                        print(f"An error occurred with {file_path}: {e}")

if __name__ == "__main__":
    # Set the path to your Python repo
    repo_path = '/Users/aprilyang/Desktop/CMU/24fall/11785/project/llm-robustness-experiment/pb_results/results'  # Replace with your repository path
    # Set the output text file
    output_file = 'results.txt'
    # You can add other file extensions if needed
    file_extensions = ['.json']  # Only Python files

    # Run the function
    copy_code_to_txt(repo_path, output_file, file_extensions)
    print(f"Python code and file contents copied to {output_file}.")
