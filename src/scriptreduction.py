# Añadir librerías
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, ApertureStats, CircularAnnulus
from scipy.spatial import KDTree
from astropy.stats import mad_std
from skimage.measure import ransac
from skimage.transform import warp, EuclideanTransform
import os
from scipy.ndimage import shift
import gaiaxpy
from gaiaxpy import PhotometricSystem

# Añadir funciones

def read_set(carpeta):
    data_list = []
    texp_list = []

    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith('.fits'):
            ruta = os.path.join(carpeta, archivo)
            with fits.open(ruta) as hdu1:
                datos = hdu1[0].data
                datos = datos.astype(np.float64)
                exptime = hdu1[0].header.get('EXPTIME', None)

                data_list.append(datos)
                texp_list.append(exptime)

    return np.array(data_list), np.array(texp_list)

def show_image(data, title=""):
    vmin = np.nanpercentile(data, 1)
    vmax = np.nanpercentile(data, 99)
    plt.imshow(data, vmin=vmin, vmax=vmax, origin="upper", cmap="gray")
    plt.colorbar()
    plt.tight_layout()
    plt.title(title)
    plt.show()
    return None

def mflat(filt):
    _flats, _texp_flat = read_set(filt)
    for _j, _flat in enumerate(_flats):
        _flat = _flat - master_bias - master_dark
        _flats[_j] = _flat/np.nanmedian(_flat)
    master_flat = np.nanmedian(_flats, axis=0)
    iden = np.where(master_flat < 0.1)
    master_flat[iden] = 0.1
    return master_flat

def science(raw, flat):
    _science_raw, _texp = read_set(raw)
    science_calib = []
    for _k, _science in enumerate(_science_raw):
        _science[_science > _threshold] = np.nan
        _calib = (_science - master_bias -
                  _texp[_k]*master_dark)/(_texp[_k]*flat)
        science_calib.append(_calib)
    science_calib = np.array(science_calib)
    return science_calib

def aligned(image1, image2, fwhmimg, threshmultiplier, sep, Filter):

    bkg1 = mad_std(image1)
    bkg2 = mad_std(image2)

    daofind1 = DAOStarFinder(fwhm=fwhmimg, threshold=threshmultiplier*bkg1)
    daofind2 = DAOStarFinder(fwhm=fwhmimg, threshold=threshmultiplier*bkg2)
    sources1 = daofind1(image1)
    sources2 = daofind2(image2)

    positions1 = np.transpose((sources1['xcentroid'], sources1['ycentroid']))

    positions2 = np.transpose((sources2['xcentroid'], sources2['ycentroid']))

    tree = KDTree(positions2)
    max_sep = sep
    matches = tree.query_ball_point(positions1, r=max_sep)

    matched_pairs = []
    for i, match in enumerate(matches):
        if match:
            distances = np.linalg.norm(positions2[match]-positions1[i], axis=1)
            closest_match = match[np.argmin(distances)]
            matched_pairs.append((positions1[i], positions2[closest_match]))

    points1_matched, points2_matched = zip(*matched_pairs)
    points1_matched = np.array(points1_matched)
    points2_matched = np.array(points2_matched)

    model_robust, inliers = ransac((points1_matched, points2_matched),EuclideanTransform, min_samples=3, residual_threshold=0.5, max_trials=1000)
    aligned_image1 = warp(image1, inverse_map=model_robust.inverse, output_shape=image1.shape, order=3)
    image2 = np.nan_to_num(image2, nan=0.0)
    aligned_image1 = np.nan_to_num(aligned_image1, nan=0.0)
    stacked = np.nanmedian([aligned_image1, image2], axis=0)
    show_image(stacked, title=Filter)
    return stacked, aligned_image1

def clean_catalog(coords, fluxes, min_sep=3.0):
    """
    Si hay estrellas más cerca de 'min_sep' entre sí, se queda solo con la más brillante.
    """
    tree = KDTree(coords)
    # Encuentra grupos de estrellas demasiado cercanas
    pairs = tree.query_pairs(min_sep)
    
    # Marcamos para borrar las que sean más débiles en cada pareja
    remove_indices = set()
    for i, j in pairs:
        if i in remove_indices or j in remove_indices:
            continue
        if fluxes[i] > fluxes[j]:
            remove_indices.add(j)
        else:
            remove_indices.add(i)
            
    # Creamos la máscara "keep" (True = quedarse)
    mask = np.ones(len(coords), dtype=bool)
    mask[list(remove_indices)] = False
    
    return mask  # Devolvemos solo la máscara para aplicar a todo

def match_catalogs(coords_ref, coords_mov, tolerance):
    tree = KDTree(coords_ref)
    distances, indices = tree.query(coords_mov, distance_upper_bound=tolerance)
    has_match = distances < tolerance
    
    raw_idx_mov = np.where(has_match)[0]
    raw_idx_ref = indices[has_match]
    raw_dists = distances[has_match]

    # Ordenar por distancia para priorizar los mejores matches
    sorter = np.argsort(raw_dists)
    sorted_idx_ref = raw_idx_ref[sorter]
    sorted_idx_mov = raw_idx_mov[sorter]
    
    # Eliminar duplicados en la referencia (Muchos de B -> 1 de V)
    _, unique_mask = np.unique(sorted_idx_ref, return_index=True)
    
    final_ref = sorted_idx_ref[unique_mask]
    final_mov = sorted_idx_mov[unique_mask]
    
    coords_match_ref = coords_ref[final_ref]
    coords_match_mov = coords_mov[final_mov]
    
    print(f"Originales Ref: {len(coords_ref)} | Originales Mov: {len(coords_mov)}")
    print(f"Parejas encontradas: {len(final_ref)}")
    
    return coords_match_ref, coords_match_mov

# %%
# Sacar información básica
_base = "/home/rodrigogp/Documents/TEA/dataset"
_info = _base+"/V/NGC884_V_00001.fits"
_info = fits.open(_info)  # Usamos la función fits para abrir los datos
_baseimg = _info[0].data
info = _info[0].header  # Accedemos al header

print("Object name :", info["OBJECT"],
      "\nExposure time :", info["EXPTIME"])

_threshold = 1e8  # 60000
_saturated_pixels = _baseimg > _threshold
print('Number of saturated pixels : ', len(_saturated_pixels[0]))
#science_no_saturated = baseimg
#science_no_saturated[saturated_pixels] = np.nan

# Master Bias
_base_bias = _base+"/Bias/"
_bias_images, _texp_bias = read_set(_base_bias)
master_bias = np.nanmedian(_bias_images, axis=0)

# Master Dark
_base_dark = _base+"/Dark4s/"
_darks, _texp_dark = read_set(_base_dark)
print("TEXP : ", _texp_dark[0])
for _i, _dark in enumerate(_darks):
    _dark = _dark
    _darks[_i] = (_dark-master_bias)/(_texp_dark[_i]) * np.ones_like(_darks[0])
master_dark = np.nanmedian(_darks, axis=0)

# Master Flat
_base_flat_V = _base+"/Flat_V/"
flat_V = mflat(_base_flat_V)
_base_flat_B = _base+"/Flat_B/"
flat_B = mflat(_base_flat_B)

# Reduced Science
_base_science_V = _base+"/V/"
_science_V = science(_base_science_V, flat_V)
_base_science_B = _base+"/B/"
_science_B = science(_base_science_B, flat_B)

# %% Alineamiento solo para dos imágenes
stacked_V, _F = aligned(_science_V[0], _science_V[1], 10.0, 15.0, 50.0, "Filter V")
stacked_B, _F = aligned(_science_B[0], _science_B[1], 10.0, 9.0, 50.0, "Filter B")

# %% Calibración de flujo    REF_ImAGE = STACKED_V

stacked_V = _science_V[1]
stacked_B = _science_B[1]

_radius=100
bkgV = mad_std(stacked_V)
bkgB = mad_std(stacked_B)

_daofindV = DAOStarFinder(fwhm=5.0, threshold=30.*bkgV) #60
_daofindB = DAOStarFinder(fwhm=5.0, threshold=30.*bkgB) #40

_sourcesV = _daofindV(stacked_V)
_sourcesB = _daofindB(stacked_B)

positionsV = np.transpose((_sourcesV["xcentroid"], _sourcesV["ycentroid"]))
positionsB = np.transpose((_sourcesB["xcentroid"], _sourcesB["ycentroid"]))

_aperturesV = CircularAperture(positionsV, r=10.)
_aperturesB = CircularAperture(positionsB, r=10.)

_fluxV = ApertureStats(stacked_V, _aperturesV).sum
_fluxB = ApertureStats(stacked_B, _aperturesB).sum

mask_V = clean_catalog(positionsV, _sourcesV['flux'], min_sep=30.0)
mask_B = clean_catalog(positionsB, _sourcesB['flux'], min_sep=30.0)
positionsV = positionsV[mask_V]
positionsB = positionsB[mask_B]

_aperturesV = CircularAperture(positionsV, r=10.)
_aperturesB = CircularAperture(positionsB, r=10.)

plt.figure(figsize=[6, 6])
plt.imshow(stacked_V, cmap='gray', origin='upper', vmin=np.nanpercentile(stacked_V, 1), vmax=np.nanpercentile(stacked_V, 99))
_aperturesV.plot(color='green', lw=10, alpha=0.8)
plt.gca().invert_xaxis()
plt.show()

plt.figure(figsize=[6, 6])
plt.imshow(stacked_B, cmap='gray', origin='upper', vmin=np.nanpercentile(stacked_B, 1), vmax=np.nanpercentile(stacked_B, 99))
_aperturesB.plot(color='blue', lw=10, alpha=0.8)
plt.gca().invert_xaxis()
plt.show()

#%%

indV = [0,4,12,19,20,22,24,27,25,33]
xycenV = positionsV[indV].astype(int)
starsV = [stacked_V[y-_radius:y+_radius, x-_radius:x+_radius] for x, y in xycenV]

apertureV = CircularAperture(xycenV, r=30)

apertureV_ = CircularAperture([_radius,_radius],r=30)

annulus_aperture_V = CircularAnnulus(xycenV, r_in=40, r_out=55)

annulus_aperture_V_ = CircularAnnulus([_radius,_radius], r_in=40, r_out=55)

plt.figure(figsize=[4, 4], dpi=100)
plt.title("Star V ")
plt.imshow(starsV[8], vmin=np.nanpercentile(starsV[8], 1),vmax=np.nanpercentile(starsV[8], 99), cmap='gray', origin='upper')
ap_patches = apertureV_.plot(color='red', lw=2, label='Photometry Aperture')
ann_patches = annulus_aperture_V_.plot(color='blue', lw=2, label='Background annulus')

plt.figure(figsize=[10,10],dpi=100)
plt.imshow(stacked_V,vmin=np.nanpercentile(stacked_V,1),vmax=np.nanpercentile(stacked_V,99),cmap='gray',origin='upper')
apertureV = CircularAperture(xycenV,r=100)
ap_patches = apertureV.plot(color='green',lw=2,label='Photometry Aperture')
#ap_patches = annulus_aperture_V.plot(color='green',lw=2,label='Photometry Aperture')
plt.gca().invert_xaxis()
plt.show()

#%% Luego para B

indB = [0,1,2,4,5,6,7,9,8,10]
xycenB = positionsB[indB].astype(int)
starsB = [stacked_B[y-_radius:y+_radius, x-_radius:x+_radius] for x, y in xycenB]

apertureB = CircularAperture(xycenB, r=30)

apertureB_ = CircularAperture([_radius,_radius],r=30)

annulus_aperture_B = CircularAnnulus(xycenV, r_in=40, r_out=55)

annulus_aperture_B_ = CircularAnnulus([_radius,_radius], r_in=40, r_out=55)

#plt.figure(figsize=[4, 4], dpi=100)
#plt.title("Star B")
#plt.imshow(starsB[0], vmin=np.nanpercentile(starsB[0], 1),vmax=np.nanpercentile(starsB[0], 99), cmap='gray', origin='upper')
#ap_patches = apertureB_.plot(color='red', lw=2, label='Photometry Aperture')
#ann_patches = annulus_aperture_B_.plot(color='blue', lw=2, label='Background annulus')

plt.figure(figsize=[10,10],dpi=100)
plt.imshow(stacked_B,vmin=np.nanpercentile(stacked_B,1),vmax=np.nanpercentile(stacked_B,99),cmap='gray',origin='upper')
#apertureB = CircularAperture(xycenB,r=100)
ap_patches = apertureB.plot(color='blue',lw=2,label='Photometry Aperture')
plt.gca().invert_xaxis()
plt.show()

#%%

src_list = [458463951857361024,
            458414542551436672,
            458461576727427712,
            458456358355233920,
            458407670603899648,
            458407601884442496,
            458459691249627776,
            458454606008605312,
            458454606008607232,
            458453334698292096] 

# Generar fotometría en sistema Johnson
photometry = gaiaxpy.generate(src_list, photometric_system=PhotometricSystem.JKC)

# El resultado contendrá la columna 'Johnson_B_mag'
print(photometry)

#%%

B_tab = photometry['Jkc_mag_B'].values
V_tab = photometry['Jkc_mag_V'].values

#%%

apertureV = CircularAperture(xycenV, r=30)
apertureB = CircularAperture(xycenB, r=30)

bgVcal = ApertureStats(stacked_V, annulus_aperture_V).median
bgBcal = ApertureStats(stacked_B, annulus_aperture_B).median
fluxVcal = ApertureStats(stacked_V, apertureV).sum
fluxBcal = ApertureStats(stacked_B, apertureB).sum
star_fluxVcal = fluxVcal - apertureV.area * bgVcal
star_fluxBcal = fluxBcal - apertureB.area * bgBcal
minstVcal = -2.5 * np.log10(star_fluxVcal)
minstBcal = -2.5 * np.log10(star_fluxBcal)

#%%
C_V_i = V_tab - minstVcal
C_B_i = B_tab - minstBcal
C_V1 = np.median(C_V_i)
C_B1 = np.median(C_B_i)
C_V2 = np.mean(C_V_i)
C_B2 = np.mean(C_B_i)
sig_C_V = np.std(C_V1)
sig_C_B = np.std(C_B1)

print("The constant of the V band will be: ", np.round(C_V1,4)," +/- ",np.round(sig_C_V,4))
print("The constant of the B band will be: ", np.round(C_B1,4)," +/- ",np.round(sig_C_B,4))

#%% ColorInd

ColIndCal1 = B_tab - V_tab
ColIndCal2 = (minstBcal + C_B1 ) - (minstVcal + C_V1)
ColIndCal3 = (minstBcal + C_B2 ) - (minstVcal + C_V2)

delta1 = ColIndCal1 - ColIndCal2
delta2 = ColIndCal1 - ColIndCal3

delta1sc = np.mean(ColIndCal1 - ColIndCal2)
delta2sc = np.mean(ColIndCal1 - ColIndCal3)

#Pues tiene pinta de que nos quedamos con C_V2 y C_B2

# %% Habría que definir un apertures 2 

#stacked_Brot = rotate(stacked_B, angle=5.0, reshape=False, cval=0.0)
stacked_B2 = shift(stacked_B, (100, 50), cval=0.0)
stacked_B2[stacked_B2 < 0] = 0
show_image(stacked_B2, title="Shifted B")

#%%

_daofindV2 = DAOStarFinder(fwhm=5.0, threshold=22.0*bkgV)
_daofindB2 = DAOStarFinder(fwhm=5.0, threshold=14.*bkgB)

_sourcesV2 = _daofindV2(stacked_V)
_sourcesB2 = _daofindB2(stacked_B2)

positionsV2 = np.transpose((_sourcesV2["xcentroid"], _sourcesV2["ycentroid"]))
positionsB2 = np.transpose((_sourcesB2["xcentroid"], _sourcesB2["ycentroid"]))

aperturesV2 = CircularAperture(positionsV2, r=10.)
aperturesB2 = CircularAperture(positionsB2, r=10.)

_fluxV2 = ApertureStats(stacked_V, aperturesV2).sum
_fluxB2 = ApertureStats(stacked_B2, aperturesB2).sum

mask_V2 = clean_catalog(positionsV2, _sourcesV2['flux'], min_sep=30.0)
mask_B2 = clean_catalog(positionsB2, _sourcesB2['flux'], min_sep=30.0)
positionsV2 = positionsV2[mask_V2]
positionsB2 = positionsB2[mask_B2]

aperturesV2 = CircularAperture(positionsV2, r=30.)
aperturesB2 = CircularAperture(positionsB2, r=30.)

plt.figure(figsize=[6, 6])
plt.imshow(stacked_V, cmap='gray', origin='upper', vmin=np.nanpercentile(stacked_V, 1), vmax=np.nanpercentile(stacked_V, 99))
aperturesV2.plot(color='green', lw=2, alpha=0.8)
plt.show()

plt.figure(figsize=[6, 6])
plt.imshow(stacked_B2, cmap='gray', origin='upper', vmin=np.nanpercentile(stacked_B2, 1), vmax=np.nanpercentile(stacked_B2, 99))
aperturesB2.plot(color='blue', lw=2, alpha=0.8)
plt.show()

#%%

###INTRODUCIR AQUÍ EL LIMPIADO DE FALSOS POSITIVOS PARA UNA MISMA ESTRELLA

# Ejecutamos el filtrado con una tolerancia de, por ejemplo, 3 píxeles
# (Si el alineado es bueno, 3px es generoso. Si es perfecto, con 1px sobra).
positionsV3, positionsB3 = match_catalogs(positionsV2, positionsB2, 60.0)

aperturesV3 = CircularAperture(positionsV3, r=30.)
aperturesB3 = CircularAperture(positionsB3, r=30.)

plt.figure(figsize=[10, 10],dpi=100)
plt.imshow(stacked_V, cmap='gray', origin='upper', vmin=np.nanpercentile(stacked_V, 1), vmax=np.nanpercentile(stacked_V, 99))
aperturesV3.plot(color='green', lw=2, alpha=0.8)
plt.show()

plt.figure(figsize=[10, 10],dpi=100)
plt.imshow(stacked_B2, cmap='gray', origin='upper', vmin=np.nanpercentile(stacked_B, 1), vmax=np.nanpercentile(stacked_B, 99))
aperturesB3.plot(color='blue', lw=2, alpha=0.8)
plt.show()

#%%

annulus_aperture_V3 = CircularAnnulus(positionsV3, r_in=37, r_out=45)
annulus_aperture_B3 = CircularAnnulus(positionsB3, r_in=37, r_out=45)

xycenV3 = positionsV3.astype(int)
xycenB3 = positionsB3.astype(int)

starsV2 = [stacked_V[y-_radius:y+_radius, x-_radius:x+_radius] for x, y in xycenV3]
starsB2 = [stacked_B2[y-_radius:y+_radius, x-_radius:x+_radius] for x, y in xycenB3]

aperturesV3_ = CircularAperture([_radius,_radius],r=30)
aperturesB3_ = CircularAperture([_radius,_radius],r=30)

annulus_aperture_V = CircularAnnulus(xycenV3, r_in=37, r_out=45)
annulus_aperture_V2_ = CircularAnnulus([_radius,_radius], r_in=37, r_out=45)

annulus_aperture_B = CircularAnnulus(xycenB3, r_in=37, r_out=45)
annulus_aperture_B2_ = CircularAnnulus([_radius,_radius], r_in=37, r_out=45)

plt.figure(figsize=[4, 4], dpi=100)
plt.title("Star V ")
plt.imshow(starsB2[19], vmin=np.nanpercentile(starsV2[8], 1),vmax=np.nanpercentile(starsV2[8], 99), cmap='gray', origin='upper')
ap_patches = aperturesV3_.plot(color='red', lw=2, label='Photometry Aperture')
ann_patches = annulus_aperture_V2_.plot(color='blue', lw=2, label='Background annulus')

#%%

bgV = ApertureStats(stacked_V, annulus_aperture_V3).median
bgB = ApertureStats(stacked_B2, annulus_aperture_B3).median
total_fluxV = ApertureStats(stacked_V, aperturesV3).sum
total_fluxB = ApertureStats(stacked_B2, aperturesB3).sum
star_fluxV = total_fluxV - aperturesV3.area * bgV
star_fluxB = total_fluxB - aperturesB3.area * bgB
minstV = -2.5 * np.log10(star_fluxV)
minstB = -2.5 * np.log10(star_fluxB)

CVVect = C_V2 * np.ones(len(minstV))
CBVect = C_B2 * np.ones(len(minstB))

V = minstV + CVVect
B = minstB + CBVect

ColorInd = B - V