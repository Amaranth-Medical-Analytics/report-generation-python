import json
import os
import re
import shutil
from ast import literal_eval

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qrcode
import seaborn as sns
import userFunctions
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import gaussian_kde

#import AIReport.userFunctions as userFunctions

def postReportStats(slideName, path, projectID, datasetID, slideInfo = False, slideInfoFile = None):
    

    # config_data = userFunctions.read_config_file(config_file_path)

    # if config_data:
    #     # Access parameters from the config data
    #     slideInfo = config_data.get("slideInfo")
    #     slideInfoFile = config_data.get("slideInfoFile")


    #Input slide data
    #slideName = 'H3U07755_F2'

    #Adjust the path here to suit your code
    path = os.path.join(path, slideName)
    report = path + '/report/'
    if os.path.exists(report):
        print('')
    else:
        os.mkdir(report)

    #Input file names
    #imageInfoFile = path + '/image_info.json'
    #nucleiFile = path + '/nuclei.csv'
    #segmentContoursFile = path + '/segmentation_contours.csv'
    #hpfMITfile = path + '/hpf.csv'
    #thumbnail = path + '/thumbnail.png'
    #tbDataFile = path + '/tubule_contours.csv'
    #imageFile = path + '/thumbnail.png'
    #mimiMITfile =  path + '/mitotic.csv'
    #celltypeFile = path + '/nuclei.csv'
    #npDataFile = path + '/nuclear_pleomorphism_data.json'
    #tbDataFile = path + '/tubule_contours.csv'
    #segmentationJson = path + '/segmentation_area_stats.json'
    #celltypePerSegmentJson = path + '/segmentation_cell_wise_data.json'
    #stilsFile = path + '/stils_patchwise_cell_type_data.csv'

    #Replace with True if clinical information is present.
    #We can maybe take this as an input parameter and the slideInfo file can have an absolute path
    #slideInfo = False
    #slideInfoFile = 'TCGA_BRCA.csv'

    #Create a statsFile
    statsFile = {
        'Mitotic cells per 10 HPF': 0,
        'Mitotic cells per 10000 tumor cells': 0
    }

    #Dictionaries
    nuclei_color_map = {'CE': (0, 255, 255),
    'TIL': (165, 42, 42),
    'End': (217, 143, 15),
    'plasma': (0, 54, 178),
    'fib': (255, 255, 255),
    'Neu': (160, 32, 240),
    'Mph': (226, 229, 35),
    'nCE': (0, 250, 146),
    'CEUK': (229, 18, 159)}

    segment_color_map = {
        "SPA": (128, 128, 128),
        "CT": (0, 54, 0),
        "ST": (173, 216, 230),
        "FAT": (212, 235, 157),
        "NE": (255, 255, 0),
        "Other": (208, 235, 241),
        "Skin": (255, 42, 42),
        "BR-D": (68, 78, 172)}

    segm_label_id_map = {
        0: "SPA",
        1: "CT",
        2: "ST",
        3: "FAT",
        4: "NE",
        5: "Other",
        6: "Skin",
        7: "BR-D"}

    stils_color_map = {'CT': (22, 55, 5),
    'ST': (173,216,230),
    'iCT': (115,233,49),
    'iST': (70,110,108),
    'pST': (251,165,49),
    'ipST': (233,238,59)}

    #Page1 Image1 Thumbnail image
    # Read image metadata
    with open(path + '/image_info.json') as file:
        meta = json.load(file)

    # Scale calculation
    image = Image.open(path + '/thumbnail.png')
    resizeW, resizeH = image.size
    h = meta['h']
    ht = resizeH
    scalet = ht / h
    box_width = (2000 / meta['mpp']) * scalet

    # Overlay the scale bar on the thumbnail image
    overlay = Image.new('RGBA', (resizeW, resizeH), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box_x = resizeW - box_width
    box_y = 0
    draw.rectangle([(box_x, box_y), (resizeW, box_y + 20)], fill='white')  # Adjust dimensions and color as needed
    font = ImageFont.truetype("Gidole-Regular.ttf")
    text = '2 mm'
    _, _, text_width, text_height = draw.textbbox((0, 0), text=text, font=font)
    text_x = box_x + (box_width - text_width) // 2
    text_y = box_y + (20 - text_height) // 2
    draw.text((text_x, text_y), text, fill=(0, 0, 0, 255), font=font)  # Adjust color as needed
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    result.save(report + 'thumbnail.png')

    #Page1 Image3 Nuclei heatmap
    annot_df = pd.read_csv(path + '/nuclei.csv')
    heatmap_img = np.zeros((meta['h'], meta['w'], 3), dtype = np.uint8)
    annot_df.apply(userFunctions.draw_annotations, axis = 1, args = (heatmap_img, nuclei_color_map))[0]
    cv2.imwrite(report + 'nuclei_heatmap.png', cv2.cvtColor(cv2.resize(heatmap_img, (resizeW, resizeH)), cv2.COLOR_BGR2RGB))
    image = Image.open(report + 'nuclei_heatmap.png')

    #Overlay scale bar 
    #Here I am assuming all the images on Page1 have the same height and width.
    overlay = Image.new('RGBA', (resizeW, resizeH), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([(box_x, box_y), (resizeW, box_y + 20)], fill='white') 
    draw.text((text_x, text_y), text, fill=(0, 0, 0, 255), font = font)  # Adjust the color as needed
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    result.save(report + 'nuclei_heatmap.png')

    #Page1 Image4 segment overlay nuclei heatmap
    df = pd.read_csv(path + '/segmentation_contours.csv')
    df.apply(userFunctions.draw_annotations, axis = 1, args = (heatmap_img, segment_color_map, 200))[0]
    cv2.imwrite(report + 'segment_overlay_nuclei_heatmap.png', cv2.cvtColor(cv2.resize(heatmap_img, (resizeW, resizeH)), cv2.COLOR_BGR2RGB))

    #Overlay scale bar 
    image = Image.open(report + 'segment_overlay_nuclei_heatmap.png')
    overlay = Image.new('RGBA', (resizeW, resizeH), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([(box_x, box_y), (resizeW, box_y + 20)], fill='white') 
    draw.text((text_x, text_y), text, fill=(0, 0, 0, 255), font = font)  # Adjust the color as needed
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    result.save(report + 'segment_overlay_nuclei_heatmap.png')

    #Page1 Image2 Segment image
    #Here I am just taking Shashank's image and overlaying the scale bar. Probably needs to change
    image = Image.open(path + '/segment.png')
    overlay = Image.new('RGBA', (resizeW, resizeH), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.text((text_x, text_y), text, fill=(0, 0, 0, 255), font = font)  # Adjust the color as needed
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    result.save(report + 'segment.png')

    #Slide information tab
    if slideInfo:
        #Read the clinical information for the project with slide names as the first column
        clinical = pd.read_csv(slideInfoFile, index_col = 0)
        caseID = slideName
        caseInfo = {"ID": clinical.at[caseID, 'ID']}
        if 'Gender' in clinical.columns:
            caseInfo['Gender'] = clinical.at[caseID, 'Gender']
        else:
            caseInfo['Gender'] = 'NotAvailable'
    
        if 'Age' in clinical.columns:
            caseInfo['Age'] = clinical.at[caseID, 'Age']
        else:
            caseInfo['Age'] = 'NotAvailable'
            
        if 'ER' in clinical.columns:
            caseInfo['ER'] = clinical.at[caseID, 'ER']
        else:
            caseInfo['ER'] = 'NotAvailable'
            
        if 'PR' in clinical.columns:
            caseInfo['PR'] = clinical.at[caseID, 'PR']
        else:
            caseInfo['PR'] = 'NotAvailable'

        if 'Her2' in clinical.columns:
            caseInfo['Her2'] = clinical.at[caseID, 'Her2']
        else:
            caseInfo['Her2'] = 'NotAvailable'
            
        if 'Stage' in clinical.columns:
            caseInfo['Stage'] = clinical.at[caseID, 'Stage']
        else:
            caseInfo['Stage'] = 'NotAvailable'
            
        if 'PAM50' in clinical.columns:
            caseInfo['PAM50'] = clinical.at[caseID, 'PAM50']
        else:
            caseInfo['PAM50'] = 'NotAvailable'
    else:
        caseInfo = {"ID": slideName}
        #caseInfo['ID'] = slideName
        caseInfo['Gender'] = 'NotAvailable'
        caseInfo['Age'] = 'NotAvailable'
        caseInfo['ER'] = 'NotAvailable'
        caseInfo['PR'] = 'NotAvailable'
        caseInfo['Her2'] = 'NotAvailable'
        caseInfo['Stage'] = 'NotAvailable'
        caseInfo['PAM50'] = 'NotAvailable'

    #caseInfo['link'] = "https://amaranth-studies.vercel.app/report?project=" + projectID + "&dataset=" + datasetID + "&image=" + slideName
    #starting a fresh link just to be sure SURE
    caseInfo['link'] = "https://amaranth-studies.vercel.app/viewer?project=" + projectID + "&dataset=" + datasetID + "&image=" + slideName
    
    with open(report + 'caseInfo.json', 'w') as json_file:
        json.dump(caseInfo, json_file)

    #QRcode generation
    #Here we need 3 user inputs: project ID, dataset ID and slideName.
    #projectID = 'test'
    #datasetID = 'test'
    qr = qrcode.QRCode(
                version = 1,
                error_correction = qrcode.constants.ERROR_CORRECT_L,
                box_size = 10,
                border = 4,
            )
    qr.add_data("https://amaranth-studies.vercel.app/report?projectId=" + projectID + "&datasetId=" + datasetID + "&imageId=" + slideName)
    qr.make(fit = True)
    qrCode = qr.make_image(fill_color = "black", back_color = "white")
    qrCode.save(report + 'qrCode.png')

    #HPF analysis
    #Look for the *hpf.csv
    if os.path.exists(path + '/hpf.csv'):
        HPFabsent = False
    else:
        HPFabsent = True

    if HPFabsent == False:
        #enter here only if HPFs were found in the image
        hpfMIT = pd.read_csv(path + '/hpf.csv')
        hpfMIT
        i = 0
        
        #Split the HPF file by HPF and MIT cells
        hpf = hpfMIT[hpfMIT['annot_type'] == 'HPF']
        mit = hpfMIT[hpfMIT['annot_type'] == 'MIT']
        
        #Check which mitotic cell belongs to which HPF
        for mit_index, mit_row in mit.iterrows():
            mit_coords = eval(mit.at[mit_index, 'points'].replace("{", "[").replace("}", "]"))
            mit_coords = np.array(mit_coords)

            for hpf_index, hpf_row in hpf.iterrows():
                coords = hpf_row['points']
                hpf_coords = eval(hpf.at[hpf_index, 'points'].replace("{", "[").replace("}", "]"))
                hpf_coords = np.array(hpf_coords)
                is_inside1 = np.all((mit_coords[0][0] >= hpf_coords.min(axis=0)[0]) & (mit_coords[0][0] <= hpf_coords.max(axis=0)[0]) & (mit_coords[0][1] >= hpf_coords.min(axis = 0)[1]) & (mit_coords[0][1] <= hpf_coords.max(axis = 0)[1]))       
                is_inside2 = np.all((mit_coords[2][0] >= hpf_coords.min(axis=0)[0]) & (mit_coords[2][0] <= hpf_coords.max(axis=0)[0]) & (mit_coords[2][1] >= hpf_coords.min(axis = 0)[1]) & (mit_coords[2][1] <= hpf_coords.max(axis = 0)[1]))
                
                #Checking here if top-left and bottom-right coordinates of the mitotic cell are inside the HPF
                if is_inside1 and is_inside2:
                    mit.loc[mit_index, 'HPF'] = str(hpf_index + 1)
                    mit.loc[mit_index, 'HPFcoords'] = coords

        #Remove any mitotic cells which don't belong to any HPFs
        mit = mit[mit['HPF'].notna()]
        mit = mit.sort_values(by='HPF', ascending=True)

        mit_hpf_count = mit['HPF'].value_counts()
        mit_hpf_count = pd.DataFrame(mit_hpf_count)
        mit_hpf_count = mit_hpf_count.reset_index()
        mit_hpf_count.columns = ['HPF', 'values']
        
        #Expansion_pixels is the expanding factor for mitotic cells. Currently hard coded
        expansion_pixels = 50
        
        #This loop generates HPF files which highlight the mitotic cells and also crop the mitotic cells
        for hpfNum in range(1, 11):
            hpfNum = str(hpfNum)
            subset_df = mit[mit['HPF'] == hpfNum]
            if subset_df.shape[0] == 0:
                #Enter here if no mitotic cell is present in the current HPF
                shutil.copy(path + '/hpf_' + hpfNum + '.png', report + 'hpf_' + hpfNum + '_mitosis.png')
                next
            image = Image.open(path + '/hpf_' + hpfNum + '.png')
            draw = ImageDraw.Draw(image)
            const = 20
            img = cv2.imread(path + '/hpf_' + hpfNum + '.png')

            for index, row in subset_df.iterrows():
                temp = 0
                mit_coords = eval(subset_df.at[index, 'points'].replace("{", "[").replace("}", "]"))
                hpf_coords = eval(subset_df.at[index, 'HPFcoords'].replace("{", "[").replace("}", "]"))

                subtract_x = hpf_coords[0][0]
                subtract_y = hpf_coords[0][1]
                rect_coords = [(x - subtract_x + const, y - subtract_y + const) for x, y in mit_coords]
                flattened_points = [coord for point in rect_coords for coord in point]
                draw.polygon(flattened_points, outline = 'yellow', width = 10)

                centroid_x, centroid_y = userFunctions.find_polygon_centroid(polygon_coords= rect_coords)
                vertices = [(centroid_x + expansion_pixels, centroid_y + expansion_pixels),
                            (centroid_x - expansion_pixels, centroid_y + expansion_pixels),
                            (centroid_x - expansion_pixels, centroid_y - expansion_pixels),
                            (centroid_x + expansion_pixels, centroid_y - expansion_pixels)]
                
                #check if anyone of the vertices are outside the HPF. The vertices are calculated post expansion
                if userFunctions.has_negative_coordinates(coordinates_list = vertices):
                    new_vertices = []
                    temp = 1
                    for x, y in vertices:
                        x = max(x, 0)
                        y = max(y, 0)
                        new_vertices.append((x, y))
                    vertices = new_vertices
                
                xstart, ystart = map(int, rect_coords[0])
                xend, yend = map(int, rect_coords[2])
                cv2.rectangle(img, (xstart, ystart), (xend, yend), (255, 0, 0), 2)
                int_list = [[int(num) for num in tup] for tup in vertices]
                x1, y1 = map(int, int_list[0])
                x2, y2 = map(int, int_list[2])
                cropped_image = img[y2:y1, x2:x1,]

                #Set grey background for cells which have vertices outside the HPF.
                #This is done to maintain the same cell sizes
                if temp == 1:
                    black_background = np.ones((100, 100, 3), dtype=np.uint8) * 100
                    x_offset = (black_background.shape[1] - cropped_image.shape[1]) // 2
                    y_offset = (black_background.shape[0] - cropped_image.shape[0]) // 2
                    black_background[y_offset:y_offset + cropped_image.shape[0], x_offset:x_offset + cropped_image.shape[1]] = cropped_image
                    cv2.imwrite(report + str(index) + '_HPF_' + str(hpfNum) + '.png', cv2.resize(black_background, (200, 200)))
                else:
                    cv2.imwrite(report + str(index) + '_HPF_' + str(hpfNum) + '.png', cv2.resize(cropped_image, (200, 200)))

            image.save(report + 'hpf_' + hpfNum + '_mitosis.png')

        mit1 = mit
        mit = mit.iloc[:, :5]
        #Rewrite the updated HPF file 
        hpfMIT = pd.concat([hpf, mit], ignore_index=True)
        hpfMIT.to_csv(path + '/hpf.csv', index=False)
        mit1['HPF'].value_counts()
        
        with open(path + '/image_info.json') as file:
            meta = json.load(file)
        #HPF thumbnail image
        hpf_img = np.zeros((meta['h'], meta['w'], 3), dtype=np.uint8)
        for index, row in hpf.iterrows():
            points = eval(hpf.at[index, 'points'].replace("{", "[").replace("}", "]"))
            ## convert to contour
            ctr = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            hpf_img = cv2.rectangle(hpf_img, (ctr.min(axis=0)[0][0], ctr.min(axis=0)[0][1]), (ctr.max(axis=0)[0][0], ctr.max(axis=0)[0][1]), (255,255,10), 200)

        image = Image.open(path + '/thumbnail.png')
        resizeW, resizeH = image.size
        img_color = cv2.resize(hpf_img, (resizeW, resizeH))
        mask = img_color.astype(bool)

        image = cv2.imread(report + 'thumbnail.png')
        image[mask] = cv2.addWeighted(image,.25, img_color, .75, gamma = 0)[mask]
        cv2.imwrite(report + 'hpfThumbnail.png', image)

    #Tubule formation data
    with open(path + '/image_info.json') as file:
        meta = json.load(file)
        
    df = pd.read_csv(path + '/tubule_contours.csv', index_col = [0])
    #Remove all the entries which say 'Other' in the annot_type column
    subset_df = df[df['annot_type'] != 'Other']

    image = Image.open(report + 'thumbnail.png')
    resizeW, resizeH = image.size

    #Generate the tubule contour image
    tub_img = np.zeros((meta['h'], meta['w'], 3), dtype=np.uint8)
    for index, row in subset_df.iterrows():
        points = eval(subset_df.at[index, 'points'].replace("{", "[").replace("}", "]"))
        ## convert to contour
        ctr = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        col = (10, 10, 255) if row['in_CT'] else (255, 10, 10)
        tub_img = cv2.drawContours(tub_img, [ctr], 0, col, thickness = -1)

    img_color = cv2.resize(tub_img, (resizeW, resizeH))
    mask = img_color.astype(bool)

    image = cv2.imread(report + '/thumbnail.png')
    image[mask] = cv2.addWeighted(image, 0.25, img_color, 0.75, gamma = 0)[mask]
    cv2.imwrite(report + 'tbImage.png', image)

    #Tubule stats
    with open(path + '/tubule.json') as file:
        tbInfo = json.load(file)
    tb_mm = (tbInfo['total_number_tubules_in_ct_10']/tbInfo['ct_area'])*1000000
    tbScore = userFunctions.tbCalc(tb_mm, tbInfo['total_percentage_tubule_area_in_ct'])
    tbInfo['Tubule/mm2'] = tb_mm
    tbInfo['Tubule score'] = tbScore
    with open(report + 'tbInfo.json', 'w') as json_file:
        json.dump(tbInfo, json_file)
    statsFile = {**statsFile, **tbInfo}

    #Nuclear pleomorphism data
    with open(path + '/nuclear_pleomorphism_data.json') as file:
        npData = json.load(file)
    # Extract data
    side = npData['side_data']
    vesicularity = npData['vasicularity_data']


    #Constant
    normal_mean = 8.07

    #Calculate statistics
    sideMean = mean_val = round(np.mean(side), 2)
    sideSD = std_dev = round(np.std(side), 2)
    statsFile['Side median'] = median_val = round(np.median(side), 2)
    q1 = np.percentile(side, 25)
    q3 = np.percentile(side, 75)
    sideIQR = iqr = round(q3 - q1, 2)

    #Side density plot
    plt.figure(figsize=(8, 6))  # Setting the size of the plot (optional)
    ax = sns.kdeplot(side, fill=True, color = "blue")
    ax.axvline(mean_val, linestyle = "dashed", color = 'blue')
    ax.axvline(normal_mean, linestyle = "dashed", color = 'red')
    # Adding title and labels
    plt.title("Size Density Plot", fontsize=16)
    plt.xlabel(r'$ \longrightarrow $' + 'Density')
    plt.ylabel("Density")
    plt.grid(False)
    plt.savefig(report + '/size.png', dpi = 200)

    #Add data to stats file

    statistics_table = pd.DataFrame({
        "Statistic": ["Normal mean", "Mean", "Standard Deviation", "Median", "IQR"],
        "Value": [normal_mean, mean_val, std_dev, median_val, iqr]
    })
    statistics_table.to_json(report + '/sideDensity.json', orient='index')

    #Constant
    normal_mean = 0.66

    #Calculate stats
    statsFile['vesicularity mean'] = mean_val = round(np.mean(vesicularity), 2)
    statsFile['vesicularity SD'] = std_dev = round(np.std(vesicularity), 2)
    statsFile['vesicularity median'] = median_val = round(np.median(vesicularity), 2)
    q1 = np.percentile(vesicularity, 25)
    q3 = np.percentile(vesicularity, 75)
    statsFile['vesicularity IQR'] = iqr = round(q3 - q1, 2)


    plt.figure(figsize=(8, 6))  # Setting the size of the plot (optional)
    ax = sns.kdeplot(vesicularity, fill=True, color = "blue")
    ax.axvline(mean_val, linestyle = "dashed", color = 'blue')
    ax.axvline(normal_mean, linestyle = "dashed", color = 'red')
    # Adding title and labels
    plt.title("Vesicularity Density Plot", fontsize=16)
    plt.xlabel(r'$ \longleftarrow $' + 'Vesicularity')
    plt.ylabel("Density")
    plt.grid(False)
    plt.savefig(report + '/vesicularity.png', dpi = 200)

    statistics_table = pd.DataFrame({
        "Statistic": ["Normal mean", "Mean", "Standard Deviation", "Median", "IQR"],
        "Value": [normal_mean, mean_val, std_dev, median_val, iqr]
    })
    statistics_table.to_json(report + '/vesicularityDensity.json', orient='index')

    #Nucleoli data
    with open(path + '/nucleoli_data.json') as file:
        ncData = json.load(file)
    categories = list(ncData.keys())
    frequencies = list(ncData.values())
    plt.figure(figsize=(8, 6)) # Setting the size of the plot (optional)
    plt.xlabel(r'$ \longrightarrow $' + 'Confidence')
    plt.ylabel('proportion of cells with prominent nucleoli')
    plt.bar(categories, frequencies)
    plt.ylim(0, 1)
    plt.title("Proportions of cells with prominent nucleoli", fontsize=16)
    plt.savefig(report + '/nucleoli.png', dpi = 200)

    if HPFabsent:
        print('prominent nucleoli absent')
    else:
        prominentN = pd.read_csv(path + '/prominent_cells_first_hpf.csv')
        nucleiData = pd.read_csv(path + '/nuclei.csv')
        prominent_coordinates = set(zip(prominentN['x1'], prominentN['y1']))
        def is_prominent(row):
            return (row['x1'] - 2, row['y1'] - 2) in prominent_coordinates

        # Apply the function to each row and create the 'prominent' column
        nucleiData['prominent'] = nucleiData.apply(is_prominent, axis=1)

        # Count the values in the 'prominent' column
        value_counts = nucleiData['prominent'].value_counts()

        #Assign a column named prominent in the HPF file to check.
        hpf = pd.read_csv(path + '/hpf.csv')
        hpf_coords = eval(hpf.at[0, 'points'].replace("{", "[").replace("}", "]"))
        hpf_coords1 = np.array(hpf_coords)
        subset_df = nucleiData

        #Also find cells from the nucleiData df which lie within HPF1
        for ce_index, ce_row in subset_df.iterrows():
            coords = ce_row['points']
            values = re.findall(r'\d+', coords)
            data_list = [int(value) for value in values]
            mit_coords = np.array(data_list).reshape(-1, 2)
            is_inside = np.all((mit_coords[0][0] >= hpf_coords1.min(axis=0)[0]) & (mit_coords[0][0] <= hpf_coords1.max(axis=0)[0]) & (mit_coords[0][1] >= hpf_coords1.min(axis = 0)[1]) & (mit_coords[0][1] <= hpf_coords1.max(axis = 0)[1]))

            if is_inside:
                subset_df.at[ce_index, 'HPF'] = 'Keep'

        ce = subset_df[subset_df['HPF'] == 'Keep']
        ce['prominent'].value_counts()

        image = Image.open(path + '/hpf_1.png')
        draw = ImageDraw.Draw(image)
        const = 20

        for index, row in ce.iterrows():
            mit_coords = eval(ce.at[index, 'points'].replace("{", "[").replace("}", "]"))
            subtract_x = hpf_coords[0][0]
            subtract_y = hpf_coords[0][1]
            rect_coords = [(x - subtract_x + const, y - subtract_y + const) for x, y in mit_coords]
            flattened_points = [coord for point in rect_coords for coord in point]
            if ce.at[index, 'prominent']:
                col = (255, 255, 0)
            else:
                col = nuclei_color_map[ce.at[index, 'annot_type']]
            draw.polygon(flattened_points, outline = col, width = 3)

        image.save(report + 'hpf_npCE.png')

    #Calculate nuclear pleomorphism AI score
    confidence = ncData['0.9']
    npScore = userFunctions.npScoreCalc(sideMean, sideIQR, confidence)
    statsFile = {**statsFile, **ncData}

    with open(report + 'nucleoli_data.json', "w") as json_file:
        json.dump(ncData, json_file)

    #Page1 Table1
    columns = ['', 'AI score', 'AI translated score', 'Pathologist score']
    rows = ['Mitotic score', 'Nuclear pleomorphism', 'Glandular (Acinar)/ Tubular Differentiation', 'Overall Grade']

    # Create the DataFrame
    table1 = pd.DataFrame(columns = columns, index = rows)
    mitInfo = userFunctions.MITcalc(path + '/mitotic.csv', path + '/hpf.csv', path + '/nuclei.csv')
    mitInfo_cellperCE = int(round(mitInfo[1], 0))

    if HPFabsent:
        mitScore = 1
        
        #Will enter here only if HPFs were not found.
        #Also the values here are hard coded
        if mitInfo_cellperCE > 20:
            mitScore = 2
        if mitInfo_cellperCE > 40:
            mitScore = 3
    else:
        mitScore = userFunctions.mitoticScoreCalc(mitInfo[0])

    #Mitotic Score     
    table1.at['Mitotic score', 'AI score'] = str(mitInfo[0]['MIT']) + " mitotsis/ " + str(mitInfo[0]['HPF']) + " HPF\n" + str(mitInfo_cellperCE) + " mitotsis/10000 tumor cells"
    table1.at['Mitotic score', 'AI translated score'] = mitScore

    #Nuclear pleomorphism score
    table1.at['Nuclear pleomorphism', 'AI score'] = str(round(sideMean, 1)) + " \u00b5m\u00b2 mean, " + str(round(sideSD, 1)) + "  std"
    table1.at['Nuclear pleomorphism', 'AI translated score'] = npScore

    #Tubule formation score
    table1.at['Glandular (Acinar)/ Tubular Differentiation', 'AI score'] = str(tbInfo['total_number_tubules_in_ct_10']) + " glandular/tubular structures"
    table1.at['Glandular (Acinar)/ Tubular Differentiation', 'AI translated score'] = tbScore

    table1.at['Overall Grade', 'AI translated score'] = mitScore + npScore + tbScore
    table1_summary = table1.fillna(' ')
    table1 = table1.fillna(' ')
    mitInfo = userFunctions.MITcalc(path + '/mitotic.csv', path + '/hpf.csv', path + '/nuclei.csv')
    mitScore = userFunctions.mitoticScoreCalc(mitInfo[0])
    mitInfo_cellperCE = int(round(mitInfo[1], 0))
    table1_summary.to_json(report + 'table1_summary.json', orient='index')


    statsFile['Mitotic cells per 10 HPF'] = mitInfo[0]['MIT']
    statsFile['Mitotic cells per 10000 tumor cells'] = mitInfo_cellperCE
    statsFile['Side mean'] = sideMean
    statsFile['Side std'] = sideSD
    statsFile['Side IQR'] = sideIQR
    statsFile['mitoticScore'] = mitScore
    statsFile['npScore'] = npScore

    #Page1 Table2
    table2 = userFunctions.segmentationTable(path + '/segmentation_area_stats.json')
    table2.to_json(report + 'table2_summary.json', orient='index')
    totalArea = table2['area in mm2'].sum()

    table2.reset_index(inplace=True)
    melted_df = pd.melt(table2, id_vars=['index'], var_name='Attribute', value_name='Value')
    # Combine 'index' and 'Attribute' into a single column
    melted_df['Combined'] = melted_df['index'] + ' ' + melted_df['Attribute']
    # Drop the 'index' and 'Attribute' columns
    melted_df.drop(columns=['index', 'Attribute'], inplace=True)
    # Rename columns for clarity
    melted_df.columns = ['Value', 'Segment']
    melted_df = melted_df[['Segment', 'Value']]
    result_dict = result_dict = melted_df.set_index('Segment')['Value'].to_dict()
    statsFile = {**statsFile, **result_dict}

    #Page1 Table3
    table3 = userFunctions.celltypeTable(path + '/nuclei.csv', path + '/mitotic.csv', totalArea)
    table3.columns.values[0] = 'Celltype'
    table3.to_json(report + 'table3_summary.json', orient='index')
    cellTable = table3.set_index('Celltype')

    #HPF_mitTable
    columns = ['Mitotic cells']
    rows = ['Total count', 'per mm2', 'per 1000 tumor cells', 'Total mitosis in 10 consecutive HPF', 'AI derived score']

    # Create the DataFrame
    HPF_mitTable = pd.DataFrame(columns = columns, index = rows)
    HPF_mitTable.at['Total count', 'Mitotic cells'] = cellTable.at['MIT', 'Total count']
    HPF_mitTable.at['per mm2', 'Mitotic cells'] = round(cellTable.at['MIT', 'per mm2'], 1)
    HPF_mitTable.at['per 1000 tumor cells', 'Mitotic cells'] = round(cellTable.at['MIT', 'per 1000 epithelial cells'], 1)
    HPF_mitTable.at['Total mitosis in 10 consecutive HPF', 'Mitotic cells'] = mitInfo[0]['MIT']
    HPF_mitTable.at['AI derived score', 'Mitotic cells'] = mitScore
    HPF_mitTable.to_json(report + 'HPF_mitTable.json', orient='index')

    melted_df = pd.melt(table3, id_vars=['Celltype'], var_name='Attribute', value_name='Value')
    print('Testing here')
    # Combine 'index' and 'Attribute' into a single column
    melted_df['Combined'] = melted_df['Celltype'] + ' ' + melted_df['Attribute']

    # Drop the 'index' and 'Attribute' columns
    melted_df.drop(columns=['Celltype', 'Attribute'], inplace=True)

    # Rename columns for clarity
    melted_df.columns = ['Value', 'Celltype']
    melted_df = melted_df[['Celltype', 'Value']]
    result_dict = result_dict = melted_df.set_index('Celltype')['Value'].to_dict()
    statsFile = {**statsFile, **result_dict}

    #Page1 Table4
    table4 = userFunctions.celltypePerSegmentTable(path + '/segmentation_cell_wise_data.json')
    table4.to_json(report + 'table4_summary.json', orient='index')

    table4.reset_index(inplace=True)
    melted_df = pd.melt(table4, id_vars=['Celltype'], var_name='Attribute', value_name='Value')
    melted_df['Combined'] = melted_df['Celltype'] + ' ' + melted_df['Attribute']

    # Drop the 'index' and 'Attribute' columns
    melted_df.drop(columns=['Celltype', 'Attribute'], inplace=True)

    # Rename columns for clarity
    melted_df.columns = ['Value', 'Segment']
    melted_df = melted_df[['Segment', 'Value']]
    result_dict = result_dict = melted_df.set_index('Segment')['Value'].to_dict()
    statsFile = {**statsFile, **result_dict}

    #sTILs Page
    with open(path + '/image_info.json') as file:
        meta = json.load(file)

    #Generate heatmap for sTILs
    heatmap_img = np.zeros((meta['h'], meta['w'], 3), dtype = np.uint8)
    stils = pd.read_csv(path + '/stils_patchwise_cell_type_data.csv')
    stils['x'] = stils['start_x'].apply(lambda x: (x + (x + 200))/2)
    stils['y'] = stils['start_y'].apply(lambda x: (x + (x + 200))/2)
    stils['select'] = 'Keep'
    df = pd.read_csv(path + '/segmentation_contours.csv')
    #Segment from which sTILs should be removed
    values_to_check = ['FAT', 'NE']

    # Subset the DataFrame based on the condition
    df_segment = df[df['annot_type'].isin(values_to_check)]
    df_segment['contour'] = df_segment['points'].apply(lambda x: np.array(literal_eval(x)))
    stils['select'] = stils.apply(lambda row: 'remove' if userFunctions.is_centroid_inside_any_contour((row['x'], row['y']), df_segment['contour']) else row['select'], axis=1)

    stils = stils[stils['select'] == 'Keep']
    for index, row in stils.iterrows():
        x1, y1 = stils.at[index, 'start_x'], stils.at[index, 'start_y'] 
        x2, y2 = x1 + 200, y1 + 200
        color = stils_color_map[stils.at[index, 'label']]
        heatmap_img = cv2.rectangle(heatmap_img, (x1, y1), (x2, y2), color, thickness = 100)

    resizeMPP = 1.2
    resizeH = (meta['mpp'] * meta['h'])/resizeMPP
    resizeH = int(resizeH)
    resizeW = (meta['mpp'] * meta['w'])/resizeMPP
    resizeW = int(resizeW)
    cv2.imwrite(report + 'stils.png', cv2.cvtColor(cv2.resize(heatmap_img, (resizeW//20, resizeH//20)), cv2.COLOR_BGR2RGB))

    #Overlay scale bar
    mpp = meta['mpp']
    origWidth = meta['w']
    image = Image.open(report + 'stils.png')
    width, height = image.size
    box_width = (2000/mpp) * (width/origWidth)
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box_x = width - box_width
    box_y = 0
    draw.rectangle([(box_x, box_y), (width, box_y + 20)], fill= 'white')  # Adjust the dimensions and color as needed
    font = ImageFont.truetype("Gidole-Regular.ttf")
    text = '2 mm'
    _, _, text_width, text_height = draw.textbbox((0, 0), text=text, font=font)
    text_x = box_x + (box_width - text_width) // 2
    text_y = box_y + (20 - text_height) // 2
    draw.text((text_x, text_y), text, fill=(0, 0, 0, 255), font = font)  # Adjust the color as needed
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    result.save(report + 'stils.png')

    #sTILs table 1
    pixels =  200
    const = 1000
    columns = ['TILs density', 'TILs%', 'area in mm²']
    rows = ['Stromal TILs', 'Peri-tumoral TILs', 'Intra-tumoral TILs']

    # Create the DataFrame
    table = pd.DataFrame(columns = columns, index = rows)
    table

    df1 = stils[(stils['label'] == 'ST') | (stils['label'] == 'iST')]
    counts = df1['label'].value_counts()
    if 'ST' in counts.index and 'iST' in counts.index:
        val = int(counts['iST']/(counts['iST'] + counts['ST']) * 100)
        table.at['Stromal TILs', 'TILs density'] = str(round(df1['TIL_percent'].mean() * 100, 1)) + '%'
        table.at['Stromal TILs', 'TILs%'] = str(val) + '%'
        val1 = (df1.shape[0] * pixels * pixels * meta['mpp'] * meta['mpp'])/(const * const)
        table.at['Stromal TILs', 'area in mm²'] = round(val1, 2)
    else:
        table.at['Stromal TILs', 'area in mm²'] = 0


    df2 = stils[(stils['label'] == 'pST') | (stils['label'] == 'ipST')]
    counts = df2['label'].value_counts()
    if 'pST' in counts.index and 'ipST' in counts.index:
        val = int(counts['ipST']/(counts['ipST'] + counts['pST']) * 100)
        table.at['Peri-tumoral TILs', 'TILs density'] = str(round(df2['TIL_percent'].mean() * 100, 1)) + '%'
        table.at['Peri-tumoral TILs', 'TILs%'] = str(val) + '%'
        val1 = (df2.shape[0] * pixels * pixels * meta['mpp'] * meta['mpp'])/(const * const)
        table.at['Peri-tumoral TILs', 'area in mm²'] = round(val1, 2)
    else:
        table.at['Peri-tumoral TILs', 'area in mm²'] = 0


    df3 = stils[(stils['label'] == 'CT') | (stils['label'] == 'iCT')]
    counts = df3['label'].value_counts()
    if 'CT' in counts.index and 'iCT' in counts.index:
        val = int(counts['iCT']/(counts['iCT'] + counts['CT']) * 100)
        table.at['Intra-tumoral TILs', 'TILs density'] = str(round(df3['TIL_percent'].mean() * 100, 1)) + '%'
        table.at['Intra-tumoral TILs', 'TILs%'] = str(val) + '%'
        val1 = (df3.shape[0] * pixels * pixels * meta['mpp'] * meta['mpp'])/(const * const)
        table.at['Intra-tumoral TILs', 'area in mm²'] = round(val1, 2)
    else:
        table.at['Intra-tumoral TILs', 'area in mm²'] = 0

    table.to_json(report + 'stilsTable.json', orient='index')

    table.reset_index(inplace=True)
    melted_df = pd.melt(table, id_vars=['index'], var_name='Attribute', value_name='Value')
    melted_df['Combined'] = melted_df['index'] + ' ' + melted_df['Attribute']

    # Drop the 'index' and 'Attribute' columns
    melted_df.drop(columns=['index', 'Attribute'], inplace=True)

    # Rename columns for clarity
    melted_df.columns = ['Value', 'Segment']
    melted_df = melted_df[['Segment', 'Value']]
    result_dict = result_dict = melted_df.set_index('Segment')['Value'].to_dict()
    statsFile = {**statsFile, **result_dict}

    #sTILs Table2
    stils1 = stils
    label_mapping = {
        'CT': 'Intra-tumoral',
        'iCT': 'Intra-tumoral',
        'ST': 'Stromal',
        'iST': 'Stromal',
        'pST': 'Peri-tumoral',
        'ipST': 'Peri-tumoral'
    }
    stils1['label1'] = stils1['label'].replace(label_mapping)
    stils1 = stils1.drop(columns = 'label')
    stils1 = stils1.drop(columns = 'select')

    sum_by_label = stils1.groupby('label1').sum()
    sum_by_label

    columns_to_remove = ['Unnamed: 0', 'start_x', 'start_y', 'TIL_percent', 'x', 'y']
    # Removing the specified columns
    df = sum_by_label.drop(columns=columns_to_remove)
    pixels = 200
    const = 1000

    stils_counts = stils1['label1'].value_counts()
    df1 = df
    for index, row in df.iterrows():
        for column_name, value in row.items():
            denominator = stils_counts[index] * pixels * pixels * meta['mpp'] * meta['mpp']
            numerator = const * const * df.at[index, column_name]
            df1.at[index, column_name] = round(numerator/denominator, 2)

    df1.rename_axis('no. of cells per mm²', inplace=True)
    column_mapping = {
        'no_of_TIL': 'Tumor infiltrating lymphocytes',
        'no_of_End': 'Endothelial cells',
        'no_of_fib': 'Fibroblasts',
        'no_of_CE': 'Tumor/epithelial cells',
        'no_of_Neu': 'Neutrophils',
        'no_of_plasma': 'Plasma cells',
        'no_of_Mph': 'Macrophages',
        'no_of_nCE': 'Normal cells',
        'no_of_CEUK': 'Unknown cell type',
    }
    df1 = df1.rename(columns = column_mapping)
    new_index = ['Stromal', 'Peri-tumoral', 'Intra-tumoral']
    df1 = df1.reindex(new_index)
    df1.to_json(report + 'stils_celltypeTable.json', orient='index')

    df1.reset_index(inplace=True)
    melted_df = pd.melt(df1, id_vars=['no. of cells per mm²'], var_name='Attribute', value_name='Value')
    melted_df['Combined'] = melted_df['no. of cells per mm²'] + ' ' + melted_df['Attribute']

    # Drop the 'index' and 'Attribute' columns
    melted_df.drop(columns=['no. of cells per mm²', 'Attribute'], inplace=True)

    # Rename columns for clarity
    melted_df.columns = ['Value', 'Segment']
    melted_df = melted_df[['Segment', 'Value']]
    result_dict = result_dict = melted_df.set_index('Segment')['Value'].to_dict()
    statsFile = {**statsFile, **result_dict}

    finalStats = pd.DataFrame(statsFile, index=[0]).transpose()
    finalStats.to_csv(report + slideName + '_finalStats.csv', index=True)
    JSON_info = {
        'Case information' : 'caseInfo.json',
        'Summary' : 'table1_summary.json',
        'Segment model' : 'table2_summary.json',
        'Nuclei model' : 'table3_summary.json',
        'Segment/Nuclei' : 'table4_summary.json',
        'Mitotic information' : 'HPF_mitTable.json',
        'Size' : 'sideDensity.json',
        'Vesicularity' : 'vesicularityDensity.json',
        'Prominent nucleoli' : 'nucleoli_data.json', 
        'Tubule information' : 'tbInfo.json',
        'sTILs information' : 'stilsTable.json',
        'sTILs/celltype' : 'stils_celltypeTable.json'}
    with open(report + 'table_metadata.json', "w") as json_file:
        json.dump(JSON_info, json_file)

    
    mitTableFile = f'{report}HPF_mitTable.json'
    segmentTableFile = f'{report}table2_summary.json'
    nucleiTableFile = f'{report}table3_summary.json'
    nucleiSegmentTable = f'{report}table4_summary.json'
    sideFile = f'{report}sideDensity.json'
    vesicularityData = f'{report}vesicularityDensity.json'
    npDataFile = f'{report}nucleoli_data.json'

    npStats = f'{path}/nuclear_pleomorphism_data.json'
    tbFile = f'{report}tbInfo.json'

    with open(f'{report}table1_summary.json', 'r') as file:
        aiScore = json.load(file)

    mitTable = pd.read_json(mitTableFile)

    with open(segmentTableFile, 'r') as file:
        segmentData = json.load(file)
    
    total_area_mm2 = round(sum(value["area in mm2"] for key, value in segmentData.items() if key != "Other"), 1)


    with open(nucleiTableFile, 'r') as file:
        nucleiData = json.load(file)

    nucleiData = pd.read_json(nucleiTableFile)
    nucleiData.columns = nucleiData.iloc[0]
    #nucleiData

    ce = nucleiData.at['Total count', 'CE']

    ai = int(mitTable['AI derived score'][0])
    totalCount = int(mitTable['Total count'][0])
    mm2 = round(mitTable['per mm2'][0], 1)
    tumorMit = round((mitTable['Total count'][0] * 1000)/ce, 1)
    hpfMit = int(mitTable['Total mitosis in 10 consecutive HPF'][0])


    json_data = {
        "model": "Mitotic detection",
        "description": "The model helps you identify the mitotic and mitotic mimic cells in the image. \n\n [Detailed information](https://amaranth-studies.vercel.app/platform/models/13) \n\n\n\n## Abbreviations\n\n MIT: Mitotic cells\n\n mimi: Mitotic mimic cells",
        "tables": [{
            "title": f'{slideName}',
            "caption": f"AI derived score: {ai} \nTotal area analysed in mm\u00B2: {total_area_mm2}. \nTotal no. of tumor cells: {ce}",
            "columns": ['Feature', 'Mitotic cells'],
            "data": [
                {
                    'Feature': 'Total Count',
                    'Mitotic cells': totalCount
                }, 
                {
                    'Feature': 'Per mm\u00B2',
                    'Mitotic cells': mm2
                },
                {
                    'Feature': 'Per 1000 tumor cells',
                    'Mitotic cells': tumorMit
                },
                {
                    'Feature': 'Total mitosis in 10 consecutive HPF',
                    'Mitotic cells' : hpfMit
                }
            ],
        }]
    }

    mitout = f'{report}mitoticSummaryTable.json'
    with open(mitout, "w") as outfile:
        json.dump(json_data, outfile, indent = 4, default = int)  # Indent for readability

    segmentData = pd.read_json(segmentTableFile)
    segmentData = segmentData.round(1)
    segmentData.reset_index(inplace=True)
    new_column_names = list(segmentData.columns)
    new_column_names[0] = ' '
    segmentData.columns = new_column_names

    data_dict = segmentData.to_dict(orient='records')

    json_data = {
        "model": "Segmentation model",
        "description": "[Detailed information](https://amaranth-studies.vercel.app/platform/models/13) \n\n\n\n## Abbreviations\n\n BR-D: Normal/dysplastic breast\n\n CT: Cellular Tumor\n\nFAT: Fatty tissue\n\nNE: Necrosis\n\nST: Stroma",
        "tables": [{
            "title": f'{slideName}',
            "caption": f'Total area analyzed: {total_area_mm2} mm\u00B2',
            "columns": list(segmentData.columns),
            "data": data_dict
        }]
    }

    NNout = f'{report}NNsegmentSummaryTable.json'
    with open(NNout, "w") as outfile:
        json.dump(json_data, outfile, indent = 4)  # Indent for readability

    nucleiData = pd.read_json(nucleiTableFile)

    nucleiData.reset_index(inplace=True)
    nucleiData.columns = nucleiData.iloc[0]

# Drop the first row (if needed)
    nucleiData = nucleiData.drop(0)

    with open(nucleiSegmentTable, 'r') as file:
        # Load JSON data from the file
        nucleiSegmentData = json.load(file)
    
    nucleiSegmentData = pd.DataFrame(nucleiSegmentData)

    nucleiSegmentData.reset_index(inplace=True)
    nucleiSegmentData = nucleiSegmentData.rename(columns={nucleiSegmentData.columns[0]: 'Segment'})

    with open(sideFile, 'r') as file:
        # Load JSON data from the file
        sideData = json.load(file)
        
    sideData = pd.DataFrame(sideData)
    sideData.columns = sideData.iloc[0]
    sideData = sideData.transpose()
    # Drop the first row (if needed)
    #sideData = sideData.drop(0)
    sideData.reset_index(drop=True, inplace=True)

    with open(vesicularityData, 'r') as file:
        # Load JSON data from the file
        vesicularity = json.load(file)
        
    vesicularity = pd.DataFrame(vesicularity)
    vesicularity.columns = vesicularity.iloc[0]
    vesicularity = vesicularity.transpose()
    # Drop the first row (if needed)
    #sideData = sideData.drop(0)
    vesicularity.reset_index(drop=True, inplace=True)

    with open(npDataFile, 'r') as file:
        # Load JSON data from the file
        nuclearpleomorphism = json.load(file)

    nuclearpleomorphism = pd.DataFrame.from_dict(nuclearpleomorphism, orient='index', columns=['Value'])
    # np = pd.DataFrame(np)
    # np.columns = np.iloc[0]
    # np = np.transpose()
    # # Drop the first row (if needed)
    # #sideData = sideData.drop(0)
    #np.reset_index(drop=True, inplace=True)
    nuclearpleomorphism.reset_index(inplace=True)
    nuclearpleomorphism.columns = ['Confidence', 'Value']

    nuclearpleomorphismBar = nuclearpleomorphism.to_dict('records')

    with open(npStats, 'r') as file:
        npData = json.load(file)
        
    side = npData['side_data']
    data_min = min(side)
    data_max = max(side)

    # Define bin edges
    bin_edges = np.arange(data_min, data_max + 2, 1)
    bin_edges = np.round(bin_edges)
    kde = gaussian_kde(side)

    points_to_evaluate = bin_edges  # Define points to evaluate the KDE
    density_values = kde.evaluate(points_to_evaluate)


    sideplt = []
    for i in range(len(bin_edges)):
        if round(density_values[i], 2) == 0.0:
            continue
        sideplt.append({"Bin": str(round(bin_edges[i], 0)), "density": round(density_values[i], 2)})

    with open(npStats, 'r') as file:
        npData = json.load(file)
        
    vesi = npData['vasicularity_data']
    data_min = min(vesi)
    data_max = max(vesi)

    # Define bin edges
    bin_edges = np.arange(data_min, data_max + 2, 0.05)
    #bin_edges = np.round(bin_edges, 4)
    kde = gaussian_kde(vesi)

    points_to_evaluate = bin_edges  # Define points to evaluate the KDE
    density_values = kde.evaluate(points_to_evaluate)

    vesiplt = []
    for i in range(len(bin_edges)):
        if round(density_values[i], 2) == 0.00:
            continue
        vesiplt.append({"Bin": str(round(bin_edges[i], 2)), "density": round(density_values[i], 2)})

    data_dict1 = nucleiData.to_dict(orient='records')
    data_dict2 = nucleiSegmentData.to_dict(orient='records')
    data_dict3 = sideData.to_dict(orient='records')
    data_dict4 = vesicularity.to_dict(orient='records')
    data_dict5 = nuclearpleomorphism.to_dict(orient='records')

    npScore = aiScore['Nuclear pleomorphism']['AI translated score']

    json_data = {
        "model": "Nuclei detection model",
        "description": "[Detailed information](https://amaranth-studies.vercel.app/platform/models/13) \n\n\n\n## Abbreviations\n\n TIL: Tumor infiltrating lymphocytes\n\n fib: Fibroblasts\n\nCE: Tumor cells\n\nnCE: Normal cells\n\nEnd: Endothelial cells\n\nMph: Macrophage\n\nplasma: Plasma cells\n\nNeu: Neutrophils\n\nmimi: Mitotic mimic cells\n\nMIT: Mitotic cells",
        "tables": [{
            "title": f'{slideName}',
            "caption": f'AI derived nuclear pleomorphism score: {npScore}',
            "columns": list(nucleiData.columns),
            "data": data_dict1
        },
        {
            "title": f'{slideName}',
            "caption": ' ',
            "columns": list(nucleiSegmentData.columns),
            "data": data_dict2
        },
        {
            "title": 'Cell size statistics',
            "caption": ' ',
            "columns": list(sideData.columns),
            "data": data_dict3
        },
        {
            "title": 'Cell vesicularity statistics',
            "caption": ' ',
            "columns": list(vesicularity.columns),
            "data": data_dict4
        },
        {
            "title": 'Nucleoli statistics',
            "caption": ' ',
            "columns": list(nuclearpleomorphism.columns),
            "data": data_dict5
        }],
        "plots": [{
            "title": "Nuclear Pleomorphism",
            "caption": " ",
            "type": "bar",
            "data": nuclearpleomorphismBar,
            "xAxis": {
                "title": "Confidence",
                "dataKey": "Confidence"
            },
            "yAxis": {
                "title": "Value"
            },
            "mode": "group",
            "legend": {
                "title": "Legend"
            },
            "showLegend": "true",
            "traces": [
                {
                "color": "#8884d8",
                "dataKey": "Value",
                "name": "Value"
                }
            ]
        },
        {
            "title": "Cell size statistics",
            "caption": "Area chart in recharts",
            "type": "area",
            "data": sideplt,
            "xAxis": {
                "dataKey": "Bin",
                "title": "Cell size"
            },
            "yAxis": {
                "title": "density"
            },
            "legend": {
                "title": "Legend"
            },
            "showLegend": "true",
            "traces": [
            {
            "dataKey": "density",
            "name": "density"
            }
        ]
        },
            {
            "title": "Cell vesicularity statistics",
            "caption": "Area chart in recharts",
            "type": "area",
            "data": vesiplt,
            "xAxis": {
                "dataKey": "Bin",
                "title": "Cell vesicularity"
            },
            "yAxis": {
                "title": "density"
            },
            "legend": {
                "title": "Legend"
            },
            "showLegend": "true",
            "traces": [
            {
            "dataKey": "density",
            "name": "density"
            }
        ]
        }
        ]
    }

    NNout = f'{report}nucleiDataSummaryTable.json'

    with open(NNout, "w") as outfile:
        json.dump(json_data, outfile, indent = 4)  # Indent for readability

    with open(tbFile, 'r') as file:
        tbData = json.load(file)

    
    tbStats = np.array([['Total number in CT', tbData['total_number_tubules_in_ct_10']],
                 ['Total percent area in CT', round(tbData['total_percentage_tubule_area_in_ct'], 2)],
                 ['Tubules per mm\u00B2', round(tbData['Tubule/mm2'], 2)]])

    tb_df = pd.DataFrame(tbStats, columns=[' ', 'Tubules'])

    data_dict = tb_df.to_dict(orient='records')
    tbScore = aiScore['Glandular (Acinar)/ Tubular Differentiation']['AI translated score']


    json_data = {
        "model": "Tubule formation",
        "description": "[Detailed information](https://amaranth-studies.vercel.app/platform/models/13) \n\n\n\n## Abbreviations\n\n tubule_ct: Tubules in core tumor\n\n tubule_not_ct: Tubules outside of core tumor",
        "tables": [{
            "title": f'{slideName}',
            "caption": f'AI derived Glandular/Tubular differentiation score: {tbScore}',
            "columns": list(tb_df.columns),
            "data": data_dict
        }]
    }

    tbOut = f'{report}tubuleSummaryTable.json'
    with open(tbOut, "w") as outfile:
        json.dump(json_data, outfile, indent = 4)  # Indent for readability

    mimiVal = round(nucleiData.loc[2,'mimi'], 2)
    mitVal = round(nucleiData.loc[2,'MIT'], 2)

    if HPFabsent:
        data_dict = ''
        tot_mit_hpf_count = 0
    else:
        data_dict = mit_hpf_count.to_dict(orient='records')
        tot_mit_hpf_count = mit_hpf_count['values'].sum()
    
    
    json_data = {
        "model": "High Power Fields",
        "description": f"[Detailed information](https://amaranth-studies.vercel.app/platform/models/13)\n\nWe identify 10 High Power Field (HPF) starting with the HPF that has highest mitotic cells. We scan right and then down. We avoid areas with low tumor content and high immune infiltrate.  We have used the diameter of 0.51 mm so the total area of 2mm2.  According to the guidelines from Royal College of Pathologists (G148 HR, June 2016) the score is calculated as below.\n\nScore 1: up to 7\n\nScore 2: 8 - 14\n\nScore 3: more than 15\n\n**Total mitotic cells per mm\u00B2: {mitVal}**\n\n**Total mitotic mimic cells per mm\u00B2: {mimiVal}**\n\n**Total mitotic cells in 10 HPFs: {tot_mit_hpf_count}**",
        "tables":[{
            "title": 'Mitotic cells identified per HPF',
            "caption": ' ',
            "columns": list(mit_hpf_count.columns),
            "data": data_dict
        }]
    }

    hpfOut = f'{report}HPFSummaryTable.json'
    with open(hpfOut, "w") as outfile:
        json.dump(json_data, outfile, indent = 4)  # Indent for readability

    tilsFile1 = f'{report}stilsTable.json'
    tils1 = pd.read_json(tilsFile1)
    tils1.reset_index(inplace=True)
    tils1 = tils1.rename(columns={tils1.columns[0]: 'Features'})

    tilsFile2 = f'{report}stils_celltypeTable.json'
    tils2 = pd.read_json(tilsFile2)
    tils2.reset_index(inplace=True)
    tils2 = tils2.rename(columns={tils2.columns[0]: 'Celltypes'})

    data_dict1 = tils1.to_dict(orient='records')
    data_dict2 = tils2.to_dict(orient='records')

    json_data = {
        "model": "Tumor infiltrating lymphocytes",
        "description": "[Detailed information](https://amaranth-studies.vercel.app/platform/models/13) \n\n\n\n## Abbreviations\n\n CT: Core tumor\n\n iCT: Core tumor + lymphocytes\n\npST: Peri-tumoral Stroma\n\nipST: Peri-tumoral Stroma + lymphocytes\n\nST: Stroma\n\niST: Stroma + lymphocytes\n\n",
        "tables": [{
            "title": f'{slideName}',
            "caption": ' ',
            "columns": list(tils1.columns),
            "data": data_dict1
        },
        {
            "title": f'{slideName}',
            "caption": ' ',
            "columns": list(tils2.columns),
            "data": data_dict2
        }]
    }

    tilOut = f'{report}TILsummaryTable.json'
    with open(tilOut, "w") as outfile:
        json.dump(json_data, outfile, indent = 4)  # Indent for readability
