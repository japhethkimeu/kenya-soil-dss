/**
 * Step 2 - Covariate stack for Kakamega + Trans Nzoia (GEE Code Editor, JavaScript)
 *
 * Starts two exports in the Tasks tab:
 *   1. cov_kak_tnz_100m.tif         - 30-band covariate stack, 100 m, UTM 36N
 *   2. cov_kak_tnz_100m_bands.csv   - band order (GeoTIFF exports don't keep names)
 * Download both from Google Drive to data/covariates/.
 *
 * Covariates (SCORPAN):
 *   relief   elev, slope, aspect_sin/cos, tpi300, tpi2000, twi
 *   climate  rain_mm (CHIRPS 1991-2020), tmean_x10, temp_seas, rain_seas (WorldClim),
 *            aridity_idx (TerraClimate P/PET), lst_day, lst_night (MODIS)
 *   surface  Sentinel-2 median bands (all-year + Dec-Feb dry season),
 *            ndvi, ndvi_p10, ndvi_p90, bsi (bare soil index)
 */

// ------------------------------------------------------------ settings
var BBOX = [34.05, -0.15, 35.50, 1.35];   // lon_min, lat_min, lon_max, lat_max
var SCALE = 100;
var CRS = 'EPSG:32636';                   // UTM 36N covers both counties
var NODATA = -9999;
var DRIVE_FOLDER = 'soil_dss_covariates';
var PREFIX = 'cov_kak_tnz_100m';

// Optional: use a county boundary asset instead of the box.
// var aoi = ee.FeatureCollection('projects/<your-project>/assets/kenya_counties')
//   .filter(ee.Filter.inList('COUNTY', ['Kakamega', 'Trans Nzoia']))
//   .geometry().buffer(5000);
var aoi = ee.Geometry.Rectangle(BBOX);

// ------------------------------------------------------------ relief
var dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation').rename('elev');
var terr = ee.Terrain.products(dem);
var slope = terr.select('slope');
var asp = terr.select('aspect').multiply(Math.PI / 180);
var tpi300 = dem.subtract(dem.focalMean(300, 'circle', 'meters')).rename('tpi300');
var tpi2k = dem.subtract(dem.focalMean(2000, 'circle', 'meters')).rename('tpi2000');

// Topographic wetness index from MERIT Hydro upstream area (km2 -> m2, 90 m cells)
var upa = ee.Image('MERIT/Hydro/v1_0_1').select('upa').multiply(1e6);
var slopeRad = slope.multiply(Math.PI / 180).max(0.001);
var twi = upa.divide(90).divide(slopeRad.tan()).log().rename('twi');

var relief = ee.Image.cat([
  dem, slope,
  asp.sin().rename('aspect_sin'), asp.cos().rename('aspect_cos'),
  tpi300, tpi2k, twi
]);

// ------------------------------------------------------------ climate
var rain = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate('1991-01-01', '2021-01-01').sum().divide(30).rename('rain_mm');

var wc = ee.Image('WORLDCLIM/V1/BIO').select(['bio01', 'bio04', 'bio15'])
  .rename(['tmean_x10', 'temp_seas', 'rain_seas']);

var tc = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
  .filterDate('1991-01-01', '2021-01-01');
var aridity = tc.select('pr').mean()
  .divide(tc.select('pet').mean().multiply(0.1))     // pet scale factor 0.1
  .rename('aridity_idx');

var modis = ee.ImageCollection('MODIS/061/MOD11A2').filterDate('2015-01-01', '2025-01-01');
var lstDay = modis.select('LST_Day_1km').mean().multiply(0.02).subtract(273.15).rename('lst_day');
var lstNight = modis.select('LST_Night_1km').mean().multiply(0.02).subtract(273.15).rename('lst_night');

var climate = ee.Image.cat([rain, wc, aridity, lstDay, lstNight]);

// ------------------------------------------------------------ Sentinel-2
var BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
var cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate('2019-01-01', '2025-01-01')
  .linkCollection(cs, ['cs_cdf'])
  .map(function (im) {
    return ee.Image(im.updateMask(im.select('cs_cdf').gte(0.6))
      .select(BANDS).divide(10000)
      .copyProperties(im, ['system:time_start']));
  });

var prefixed = function (pfx) {
  return BANDS.map(function (b) { return pfx + b; });
};

var med = s2.median().rename(prefixed('s2_'));

// Dry season Dec-Feb (range wraps over the year end): most bare soil signal
var dry = s2.filter(ee.Filter.calendarRange(12, 2, 'month')).median()
  .rename(prefixed('s2dry_'));

var ndviCol = s2.map(function (im) {
  return im.normalizedDifference(['B8', 'B4']).rename('ndvi');
});
var ndvi = ndviCol.median().rename('ndvi');
var ndviP = ndviCol.reduce(ee.Reducer.percentile([10, 90]))
  .rename(['ndvi_p10', 'ndvi_p90']);

var bsi = dry.expression(
  "((b('s2dry_B11') + b('s2dry_B4')) - (b('s2dry_B8') + b('s2dry_B2'))) / " +
  "((b('s2dry_B11') + b('s2dry_B4')) + (b('s2dry_B8') + b('s2dry_B2')))"
).rename('bsi');

var surface = ee.Image.cat([med, dry, ndvi, ndviP, bsi]);

// ------------------------------------------------------------ stack
var stack = ee.Image.cat([relief, climate, surface])
  .toFloat().clip(aoi).unmask(NODATA);

var bandNames = stack.bandNames();
print('Number of covariates', bandNames.size());
print('Band order', bandNames);

// ------------------------------------------------------------ quick visual check
Map.centerObject(aoi, 9);
Map.addLayer(aoi, {color: 'red'}, 'AOI', false);
Map.addLayer(stack.select('elev').updateMask(stack.select('elev').neq(NODATA)),
  {min: 1100, max: 2500, palette: ['006633', 'E5FFCC', '662A00', 'D8D8D8']}, 'Elevation');
Map.addLayer(stack.select('ndvi').updateMask(stack.select('ndvi').neq(NODATA)),
  {min: 0, max: 0.9, palette: ['brown', 'yellow', 'green']}, 'NDVI median', false);
Map.addLayer(stack.select(['s2dry_B4', 's2dry_B3', 's2dry_B2']),
  {min: 0, max: 0.25}, 'S2 dry-season RGB', false);

// ------------------------------------------------------------ exports
Export.image.toDrive({
  image: stack,
  description: PREFIX,
  folder: DRIVE_FOLDER,
  fileNamePrefix: PREFIX,
  region: aoi.bounds(),
  scale: SCALE,
  crs: CRS,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

// Band order as a CSV (band_index is 1-based, matching GeoTIFF band numbers)
var bandTable = ee.FeatureCollection(
  ee.List.sequence(0, bandNames.size().subtract(1)).map(function (i) {
    return ee.Feature(null, {
      band_index: ee.Number(i).add(1),
      band_name: bandNames.get(i)
    });
  })
);
Export.table.toDrive({
  collection: bandTable,
  description: PREFIX + '_bands',
  folder: DRIVE_FOLDER,
  fileNamePrefix: PREFIX + '_bands',
  fileFormat: 'CSV',
  selectors: ['band_index', 'band_name']
});
