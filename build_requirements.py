def parse_versioned_requirements(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    versioned_dict = {}
    for line in lines:
        line = line.strip()
        # Handle the special case where the requirement is a git link
        if "@ git+" in line:
            name = line.split()[0]
            version = line.split("@")[1]
            versioned_dict[name] = "@" + version
        elif "==" in line:
            name, version = line.split("==")
            versioned_dict[name] = version

    return versioned_dict


def generate_new_requirements(versioned_dict, file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    updated_reqs = []
    for line in lines:
        line = line.strip()
        if line.startswith("--"):  # Handle --extra-index-url and similar cases
            updated_reqs.append(line)
        elif "==" in line or "@" in line:  # Handle already versioned or git links
            updated_reqs.append(line)
        else:
            # Check if the line has extras like [ffmpeg]
            if "[" in line and "]" in line:
                package_name = line.split("[")[0]
                extras = line.split("[")[1].replace("]", "")
                if package_name in versioned_dict:
                    updated_reqs.append(
                        f"{package_name}[{extras}]=={versioned_dict[package_name]}"
                    )
                else:
                    updated_reqs.append(line)
            else:
                if line in versioned_dict:
                    updated_reqs.append(f"{line}=={versioned_dict[line]}")
                else:
                    updated_reqs.append(line)

    return updated_reqs


# Parse the versioned requirements file
versioned_dict = parse_versioned_requirements("versioned_requirements.txt")

# Generate new requirements based on unversioned requirements file
new_reqs = generate_new_requirements(versioned_dict, "requirements.txt")

# Write the results to new_requirements.txt
with open("new_requirements.txt", "w") as f:
    for req in new_reqs:
        f.write(req + "\n")

print("New requirements written to new_requirements.txt")
