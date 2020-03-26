# -*- coding: utf-8 -*-
from flask import Blueprint, render_template
from flask_googlemaps import GoogleMaps, Map
import csv
from urllib.parse import unquote
import os

blueprint = Blueprint('home', __name__)


@blueprint.route('/', methods=['GET'])
def index():
        ms = []
        ps = []
        r = csv.reader(open('app/coords.csv', 'r'))
        next(r)
        for row in r:
                ps.append('new google.maps.LatLng('+row[1]+', '+row[2]+')')
        r = csv.reader(open('app/centerLocations.csv', 'r'))
        next(r)
        for row in r:
                ms.append({
                        'icon': 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png',
                        'lng':  float(row[7]),
                        'lat':  float(row[8]),
				        'infobox': ('<div class="h3" style="font-size: 20px">'+unquote(row[1])+'</div>'+
                                    '<div>'+unquote(row[3])+', '+unquote(row[0])+', '+unquote(row[4])+', '+'</div>'+
                                    '<a href='+row[6]+'>Webpage</a>')
                })

        sndmap = Map(
                identifier="sndmap",  # for DOM element
                varname="sndmap",  # for JS object name
                lat= 39.8333333,
                lng= -98.585522,
                markers=ms,
                zoom=4,
				hasoverlay=True,
		        style="height:450px;width:100%;")

        return render_template('home/index.html', sndmap=sndmap, points=','.join(ps))

@blueprint.route('/about.html')
def about():
	sndmap = Map(
                identifier="sndmap",  # for DOM element
                varname="sndmap",  # for JS object name
                lat= 39.8333333,
                lng= -98.585522,
                zoom=4,
				hasoverlay=True,
		        style="height:450px;width:100%;")
	return render_template('home/about.html',sndmap=sndmap)
