import dns.resolver
import csv

# Function to validate the domain of an email
def is_valid_domain(email):
    domain = email.split('@')[-1]
    try:
        # Check if the domain has MX records (Mail Exchange records)
        dns.resolver.resolve(domain, 'MX')
        return True
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.Timeout):
        return False

# Input and output CSV file names
input_file = 'emails_chunk_4.csv'
output_file = 'emails_chunk_1_valid.csv'

# Read emails from the CSV file
with open(input_file, 'r') as file:
    reader = csv.reader(file)
    emails = [row[0] for row in reader]  # Assuming emails are in the first column

# Filter valid emails
valid_emails = [email for email in emails if is_valid_domain(email)]

# Save valid emails to a new CSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['email'])  # Add a header
    for email in valid_emails:
        writer.writerow([email])

print(f"Valid emails have been saved to {output_file}.")
