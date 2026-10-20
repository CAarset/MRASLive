#this is the download part
def parse_metadata(pde_names):
    meta_df = pd.read_csv("pdebench_data_urls.csv")

    pde_list = [
        "advection",
        "burgers",
        "1d_cfd",
        "diff_sorp",
        "1d_reacdiff",
        "2d_cfd",
        "darcy",
        "2d_reacdiff",
        "ns_incom",
        "swe",
        "3d_cfd",
    ]

    assert all(name.lower() in pde_list for name in pde_names), "PDE name not defined."

    meta_df["PDE"] = meta_df["PDE"].str.lower()
    return meta_df[meta_df["PDE"].isin([name.lower() for name in pde_names])]


def download_data(root_folder, pde_names):
    pde_df = parse_metadata(pde_names)

    for _, row in tqdm(pde_df.iterrows(), total=pde_df.shape[0]):
        file_path = Path(root_folder) / row["Path"]
        download_url(row["URL"], file_path, row["Filename"], md5=row["MD5"])

def download_darcy():
    root_folder = Path.cwd()
    pde_names = ["darcy"]
    # Download starten
    download_data(root_folder, pde_names)