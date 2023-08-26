window.onload = function() {
	// Enable tooltip
	// const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]')
	// const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl))

	// Date picker listeners
	var startDate = $('#startDate');
	var endDate = $('#endDate');

	startDate.on('change', function(e) {
		endDate.attr('min', startDate.val());
	});
	//
	endDate.on('change', function(e) {
		startDate.attr('max', endDate.val());
	});

	// Checker button click listener
	$('#checkReddit').on('click', function() {
		let url = $("#redditPostURL").val();
		if (url == '' || !isValidRedditURL(url)) {
			$("#redditPostURL").addClass('is-invalid');
			$("#redditPostURL").parent('div').addClass('is-invalid');
			return;
		}
		$("#redditPostURL").removeClass('is-invalid');
		$("#redditPostURL").parent('div').removeClass('is-invalid');

		checkRedditByUrl(url);
	});

	// Update button click listener
	$('#updateDate').on('click', function() {
		searchByDate(startDate.val(), endDate.val());
	});

	// Tab click listener
	const triggerTabList = document.querySelectorAll('#navTab button')
	triggerTabList.forEach(triggerEl => {
		const tabTrigger = new bootstrap.Tab(triggerEl)
		triggerEl.addEventListener('click', event => {
			event.preventDefault();
			tabTrigger.show();
			const tabId = tabTrigger._config.target;
			if (tabId.indexOf('wordCloud') != -1 && misWords == null && facWords == null) {
				// $("#misLimit").attr("max", totalMisCount);
				loadWordCloud();
			} else if (tabId.indexOf('wordCloud') != -1 && misSimulation == null && facSimulation == null) {
				loadSocialNet();
			}
		})
	})

	searchByDate('', '');
}

window.onresize = function() {
	updateWordCloud(misWords, '#wordCloudMis');
	updateWordCloud(facWords, '#wordCloudFac');

	// updateSocialGraph("#socialNetMis");
	// updateSocialGraph("#socialNetFac");
}


function checkRedditByUrl(url) {
	$.post('/dashboard/classifier', {
		url: url
	}, function(data) {
		console.log(data);
	})
}


let lineChart = null;
// Specify the color scale.
const color = d3.scaleOrdinal(d3.schemeTableau10);

// Posts Trend (Line chart)
function initLineChart(mis, fac, labels) {
	const ctx = document.getElementById('lineChart');
	const misData = {
		label: 'Misinformation',
		data: mis,
		parsing: {
			xAxisKey: 'x',
			yAxisKey: 'y'
		},
		borderColor: color(0)
	};
	const facData = {
		label: 'Factual',
		data: fac,
		parsing: {
			xAxisKey: 'x',
			yAxisKey: 'y'
		},
		borderColor: color(1)
	};
	const data = {
		labels: labels,
		datasets: [misData, facData]
	};
	const config = {
		type: 'line',
		data: data,
		options: {
			scales: {
				x: {
					grid: {
						display: false
					},
				}
			},
			elements: {
				point: {
					radius: 1,
					pointStyle: 'circle',
				},
				line: {
					borderWidth: 2,
					tension: 0.3
				}
			},
			responsive: true,
			maintainAspectRatio: false,
			aspectRatio: 2.8,
			plugins: {
				legend: {
					position: 'right',
				}
			}
		}
	};
	lineChart = new Chart(ctx, config);
}

function updateLineChart(mis, fac, labels) {
	lineChart.data.labels = labels;
	lineChart.data.datasets[0].data = mis;
	lineChart.data.datasets[1].data = fac;
	lineChart.update();
}


let misWords = null;
let facWords = null;

function loadWordCloud() {
	let misLimit = 1000;
	let facLimit = 1000;
	d3.json("/dashboard/word-cloud?mis_limit="+misLimit+"&fac_limit="+facLimit, function(error, data) {
		if (error) return;
		
		misWords = data.mis;
		facWords = data.fac;
		updateWordCloud(data.mis, '#wordCloudMis');
		updateWordCloud(data.fac, '#wordCloudFac');
	});
}


function updateWordCloud(wordList, svgSelector) {
	// set the dimensions and margins of the graph
	var margin = {
			top: 10,
			right: 10,
			bottom: 10,
			left: 10
		},
		width = $(svgSelector).parent().width() - margin.left - margin.right,
		height = $(svgSelector).parent().height() - margin.top - margin.bottom;

	// append the svg object to the body of the page
	$(svgSelector).empty();
	var svg = d3.select(svgSelector)
		.attr("width", width + margin.left + margin.right)
		.attr("height", height + margin.top + margin.bottom)
		.append("g")
		.attr("transform",
			"translate(" + margin.left + "," + margin.top * 2 + ")");

	// Constructs a new cloud layout instance. It run an algorithm to find the position of words that suits your requirements
	var layout = d3.layout.cloud()
		.size([width, height])
		.words(wordList)
		.padding(5)
		// .spiral('archimedean')
		.rotate(function() {
			return ~~(Math.random() * 2) * 90;
		})
		.fontSize(d => d.size)
		.on("end", draw);
	layout.start();

	// This function takes the output of 'layout' above and draw the words
	// Better not to touch it. To change parameters, play with the 'layout' variable above
	function draw(words) {
		svg.append("g")
			.attr("transform", "translate(" + layout.size()[0] / 2 + "," + layout.size()[1] / 2 + ")")
			.selectAll("text")
			.data(words)
			.enter().append("text")
			.style("font-size", function(d) {
				return d.size + "px";
			})
			.style('fill', (d, i) => color(i % 10))
			.attr("text-anchor", "middle")
			.style("font-family", "Impact")
			.attr("transform", function(d) {
				return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
			})
			.text(function(d) {
				return d.text;
			});
	}
}


let totalMisCount = 0;
let totalFacCount = 0;

/**
 * @param {string} startDate start date in a form of 'yyyy-MM'
 * @param {string} endDate end date in a form of 'yyyy-MM'
 */
function searchByDate(startDate, endDate) {
	$("#lineChartLoading").removeClass("d-none");
	$("#lineChart").addClass("d-none");
	const maxLen = $('#lineChartLoading').height();
	$.get("/dashboard/countRedditByDate?start_date=" + encodeURIComponent(startDate) + "&end_date=" +
		encodeURIComponent(
			endDate),
		function(data) {
			$("#lineChartLoading").addClass("d-none");
			$("#lineChart").removeClass("d-none");
			const mis_final_idx = data['mis'].length - 1;
			const fac_final_idx = data['fac'].length - 1;
			let mis_x = [];
			let fac_x = [];
			data['mis'].forEach(item => {
				mis_x.push(item['x']);
			});
			data['fac'].forEach(item => {
				fac_x.push(item['x']);
			});
			// Concatenate two x label list without duplicates
			let labels = Array.from(new Set([...mis_x, ...fac_x])).sort();
			if (lineChart == null) {
				const minDate = labels[0];
				const maxDate = labels[labels.length - 1]
				$('#startDate').val(minDate);
				$('#startDate').attr('min', minDate);
				$('#startDate').attr('max', maxDate);
				$('#endDate').val(maxDate);
				$('#endDate').attr('min', minDate);
				$('#endDate').attr('max', maxDate);
				$('#lineChart').css('max-height', maxLen);
				$('#totalMisPost').text(data['misCount']);
				$('#totalFacPost').text(data['facCount']);
				// totalMisCount = data['misCount'];
				// totalFacCount = data['facCount'];
				initLineChart(data['mis'], data['fac'], labels);
			} else {
				updateLineChart(data['mis'], data['fac'], labels);
			}
		}, "json");
	return false;
}


let misSimulation = null;
let facSimulation = null;

// Load social network of posts
function loadSocialNet() {
	d3.json("/dashboard/graph?limit=200", function(error, graph) {
		if (error) return;

		misSimulation = initSocialGraph(graph.nodes.mis, graph.links.mis, "#socialNetMis");
		facSimulation = initSocialGraph(graph.nodes.fac, graph.links.fac, "#socialNetFac");
	});
}

/**
 * Initiate the d3-force graph
 * 
 * @param {Object} nodes
 * @param {Object} links
 * @param {string} svgSelector
 */
function initSocialGraph(nodes, links, svgSelector) {
	// set the dimensions and margins of the graph
	var width = $('#wordCloudMis').parent().width(),
		height = $('#wordCloudMis').parent().height();

	var tooltip = d3.select("#netTooltip")
		.style("opacity", 0);

	// Create a simulation with several forces.
	const simulation = d3.forceSimulation(nodes)
		.force("link", d3.forceLink(links).id(d => d.id))
		.force("charge", d3.forceManyBody())
		.force("x", d3.forceX())
		.force("y", d3.forceY());

	// Create the SVG container.
	const svg = d3.select(svgSelector)
		.attr("width", width)
		.attr("height", height)
		.attr("viewBox", [-width / 2, -height / 2, width, height])
		.attr("style", "max-width: 100%; height: auto;");

	// Add a line for each link, and a circle for each node.
	const link = svg.append("g")
		.attr("stroke", "#999")
		.attr("stroke-opacity", 0.6)
		.selectAll("line")
		.data(links)
		.enter()
		.append("line")
		// .join("line")
		.attr("stroke-width", d => Math.sqrt(2));

	const node = svg.append("g")
		.attr("stroke", "#fff")
		.attr("stroke-width", 1.5)
		.selectAll("circle")
		.data(nodes)
		.enter()
		.append("g")
		.append("circle")
		// .join("circle")
		.attr("r", 5)
		.attr("fill", d => color(d.label))
		.on("mouseover", function(d, i) {
			// console.log(d);
			d3.select(this).transition()
				.duration('50')
				.attr('opacity', '.85');
			// Set the tooltip content
			let label = d.label + ": " + d.id;
			tooltip.html(label)
				.style("left", (d3.event.pageX + 12) + "px")
				.style("top", (d3.event.pageY - 15) + "px");
			// Make the tooltip appear on hover
			tooltip.transition()
				.duration(50)
				.style("opacity", 1);
		})
		.on("mouseout", function() {
			d3.select(this).transition()
				.duration('50')
				.attr('opacity', '1');
			// Make the tooltip disappear
			tooltip.transition()
				.duration(50)
				.style("opacity", 0);
		});

	const label = node.append("g")
		.append("text")
		.attr('x', 6)
		.attr('y', 3)
		.text(d => d.label + ": " + d.id);

	// Add a drag behavior.
	node.call(d3.drag()
		.on("start", dragstarted)
		.on("drag", dragged)
		.on("end", dragended));

	// Set the position attributes of links and nodes each time the simulation ticks.
	simulation.on("tick", () => {
		link
			.attr("x1", d => d.source.x)
			.attr("y1", d => d.source.y)
			.attr("x2", d => d.target.x)
			.attr("y2", d => d.target.y);

		node
			.attr("cx", d => d.x)
			.attr("cy", d => d.y);

		// label
		// 	.attr("x", function(d) { return d.x + 8; })
		// 	.attr("y", function(d) { return d.y; });
	});

	// Reheat the simulation when drag starts, and fix the subject position.
	function dragstarted(d) {
		if (!d3.event.active) simulation.alphaTarget(0.3).restart();
		d.fx = d.x;
		d.fy = d.y;
	}

	function dragged(d) {
		d.fx = d3.event.x;
		d.fy = d3.event.y;
	}

	function dragended(d) {
		if (!d3.event.active) simulation.alphaTarget(0);
		d.fx = null;
		d.fy = null;
	}

	// When this cell is re-run, stop the previous simulation. (This doesn’t
	// really matter since the target alpha is zero and the simulation will
	// stop naturally, but it’s a good practice.)
	// invalidation.then(() => simulation.stop());

	return simulation;
}


function updateSocialGraph(simulation, svgSelector) {
	// set the dimensions and margins of the graph
	var width = $('#wordCloudMis').parent().width(),
		height = $('#wordCloudMis').parent().height();

	d3.select(svgSelector)
		.attr("width", width)
		.attr("height", height)
		.attr("viewBox", [-width / 2, -height / 2, width, height]);

	simulation.alphaTarget(0).restart();
}


function isValidRedditURL(url) {
	let exp = /^https:\/\/[a-z.]*redd/;
	return exp.test(url);
}


Date.prototype.format = function(fmt) {
	var o = {
		"M+": this.getMonth() + 1, //月份
		"d+": this.getDate(), //日
		"h+": this.getHours(), //小时
		"m+": this.getMinutes(), //分
		"s+": this.getSeconds(), //秒
		"q+": Math.floor((this.getMonth() + 3) / 3), //季度
		"S": this.getMilliseconds() //毫秒
	};
	if (/(y+)/.test(fmt)) {
		fmt = fmt.replace(RegExp.$1, (this.getFullYear() + "").substr(4 - RegExp.$1.length));
	}
	for (var k in o) {
		if (new RegExp("(" + k + ")").test(fmt)) {
			fmt = fmt.replace(RegExp.$1, (RegExp.$1.length == 1) ? (o[k]) : (("00" + o[k]).substr(("" + o[
					k])
				.length)));
		}
	}
	return fmt;
}
