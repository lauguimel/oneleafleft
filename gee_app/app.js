/*
 * Deforestation Forecast — Congo Basin
 * Google Earth Engine App
 *
 * Displays Prithvi-EO deforestation probability maps overlaid with
 * Hansen forest loss, WDPA protected areas, and OpenStreetMap roads.
 *
 * To deploy:
 *   1. Open https://code.earthengine.google.com/
 *   2. Paste this script
 *   3. Run → Apps → New App → Publish
 *
 * Asset IDs (update after upload):
 *   users/guillaumemaitrejean/deforest/forecast_2024
 *   users/guillaumemaitrejean/deforest/forecast_2025
 *   users/guillaumemaitrejean/deforest/forecast_2026
 */

// ── Config ───────────────────────────────────────────────────────────────────
var ASSET_PREFIX = 'users/guillaumemaitrejean/deforest/forecast_';
var YEARS = [2024, 2025, 2026];
var DEFAULT_YEAR = 2026;

var STUDY_AREA = ee.Geometry.Rectangle([20, 0, 30, 10]);
var HANSEN = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');
var WDPA = ee.FeatureCollection('WCMC/WDPA/current/polygons');

// ── Colour palette ───────────────────────────────────────────────────────────
var FORECAST_VIS = {
  min: 0, max: 0.5,
  palette: ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027'],
};

var LOSS_VIS = {
  min: 0, max: 1,
  palette: ['#000000', '#ff0000'],
};

// ── UI ───────────────────────────────────────────────────────────────────────
ui.root.setLayout(ui.Panel.Layout.absolute());

var mapPanel = ui.Map();
mapPanel.setCenter(25, 5, 7);
mapPanel.setControlVisibility({layerList: true, zoomControl: true});
ui.root.clear();
ui.root.add(mapPanel);

// Sidebar
var sidebar = ui.Panel({
  style: {
    width: '320px',
    padding: '12px',
    backgroundColor: '#f8f9fa',
  },
});

sidebar.add(ui.Label('🌳 Deforestation Forecast', {
  fontSize: '20px', fontWeight: 'bold', margin: '0 0 8px 0',
}));
sidebar.add(ui.Label('Congo Basin — Prithvi-EO segmentation model', {
  fontSize: '12px', color: '#666', margin: '0 0 16px 0',
}));

// Year selector
sidebar.add(ui.Label('Forecast year:', {fontWeight: 'bold'}));
var yearSelect = ui.Select({
  items: YEARS.map(function(y) { return String(y); }),
  value: String(DEFAULT_YEAR),
  onChange: updateMap,
  style: {stretch: 'horizontal'},
});
sidebar.add(yearSelect);

// Layer toggles
sidebar.add(ui.Label('Layers:', {fontWeight: 'bold', margin: '12px 0 4px 0'}));

var showForecast = ui.Checkbox('Forecast probability', true);
var showLoss = ui.Checkbox('Hansen observed loss', false);
var showWDPA = ui.Checkbox('Protected areas (WDPA)', false);

showForecast.onChange(updateMap);
showLoss.onChange(updateMap);
showWDPA.onChange(updateMap);

sidebar.add(showForecast);
sidebar.add(showLoss);
sidebar.add(showWDPA);

// Opacity slider
sidebar.add(ui.Label('Forecast opacity:', {margin: '12px 0 4px 0'}));
var opacitySlider = ui.Slider({
  min: 0, max: 1, value: 0.7, step: 0.1,
  onChange: updateMap,
  style: {stretch: 'horizontal'},
});
sidebar.add(opacitySlider);

// Legend
sidebar.add(ui.Label('Legend:', {fontWeight: 'bold', margin: '16px 0 4px 0'}));

var legendPanel = ui.Panel({layout: ui.Panel.Layout.flow('horizontal')});
var colours = ['#1a9850', '#91cf60', '#fee08b', '#fc8d59', '#d73027'];
var labels = ['0%', '10%', '25%', '40%', '50%+'];
for (var i = 0; i < colours.length; i++) {
  legendPanel.add(ui.Panel({
    widgets: [
      ui.Label('', {backgroundColor: colours[i], width: '24px', height: '14px', margin: '0'}),
      ui.Label(labels[i], {fontSize: '10px', margin: '0 4px'}),
    ],
    layout: ui.Panel.Layout.flow('horizontal'),
  }));
}
sidebar.add(legendPanel);

// Info
sidebar.add(ui.Label(
  'Model: Prithvi-EO-2.0 (LoRA) + UPerNet\n' +
  'Resolution: 30 m\n' +
  'Target: P(deforestation in [t, t+2])\n' +
  'Labels: Hansen GFC v1.12',
  {fontSize: '11px', color: '#888', margin: '16px 0 0 0', whiteSpace: 'pre'},
));

sidebar.add(ui.Label('Contact: guillaume.maitrejean@qut.edu.au', {
  fontSize: '10px', color: '#aaa', margin: '8px 0 0 0',
}));

// Click inspector
sidebar.add(ui.Label('Click map for pixel info', {
  fontSize: '11px', color: '#666', margin: '16px 0 4px 0', fontStyle: 'italic',
}));
var clickInfo = ui.Label('', {fontSize: '11px'});
sidebar.add(clickInfo);

mapPanel.onClick(function(coords) {
  var year = parseInt(yearSelect.getValue());
  var asset = ee.Image(ASSET_PREFIX + year);
  var point = ee.Geometry.Point([coords.lon, coords.lat]);

  asset.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point,
    scale: 30,
  }).evaluate(function(result) {
    if (result) {
      var prob = result['b1'] || result['Band 1'] || 0;
      clickInfo.setValue(
        'Lon: ' + coords.lon.toFixed(4) + ', Lat: ' + coords.lat.toFixed(4) +
        '\nP(deforestation): ' + (prob * 100).toFixed(1) + '%'
      );
    }
  });
});

// Add sidebar to map
mapPanel.add(sidebar);

// ── Map update function ──────────────────────────────────────────────────────

function updateMap() {
  mapPanel.layers().reset();

  var year = parseInt(yearSelect.getValue());
  var opacity = opacitySlider.getValue();

  // Forest basemap (treecover2000 > 30%)
  var treecover = HANSEN.select('treecover2000');
  mapPanel.addLayer(
    treecover.updateMask(treecover.gte(30)),
    {min: 30, max: 100, palette: ['#d4e9c1', '#2d6a4f']},
    'Forest cover 2000',
    true, 0.4
  );

  // Forecast layer
  if (showForecast.getValue()) {
    var forecast = ee.Image(ASSET_PREFIX + year);
    mapPanel.addLayer(
      forecast.updateMask(forecast.gt(0.01)),
      FORECAST_VIS,
      'Forecast ' + year,
      true, opacity
    );
  }

  // Hansen loss
  if (showLoss.getValue()) {
    var loss = HANSEN.select('loss');
    mapPanel.addLayer(loss.updateMask(loss), LOSS_VIS, 'Hansen loss', true, 0.6);
  }

  // WDPA
  if (showWDPA.getValue()) {
    var wdpa = WDPA.filterBounds(STUDY_AREA);
    mapPanel.addLayer(
      wdpa.style({color: '#2196F3', fillColor: '#2196F388', width: 1}),
      {}, 'Protected areas', true, 0.5
    );
  }
}

// Initial render
updateMap();
