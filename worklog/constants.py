JOB_TYPE_OPTIONS = [
    "", "Service", "Repair", "Install", "Inspection", "Other"
]

STATUS_OPTIONS = [
    "", "New", "In Progress", "On Hold", "Completed", "Cancelled"
]

EXPENSE_TYPE_OPTIONS = [
    "", "Fuel", "Parking", "Parts", "Tools", "Meals", "Other"
]

# what you want to display in the UI (only shown if those columns exist)
UI_COLUMNS = [
    "job_number",
    "job_type",
    "status",
    "vehicle_description",
    "postcode",
    "expense_type",
    "customer_name",
    "site_address",
    "updated_at",
]