with auth_venues_top3 as(
	select authid, count(distinct venue.name) as num_venue
	from venue, papers, paperauths
	where venue.id = papers.venue
	and papers.id = paperauths.paperid
	group by authid
	order by num_venue desc
	limit 3)
select auth_venues_top3.authid, authors.name, venue.name, count(papers.id)
from auth_venues_top3, authors, venue, paperauths, papers
where auth_venues_top3.authid = authors.id
and paperauths.authid = auth_venues_top3.authid
and paperauths.paperid = papers.id
and papers.venue = venue.id
group by auth_venues_top3.authid, authors.name, venue.name;