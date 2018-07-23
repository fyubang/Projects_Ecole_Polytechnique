create extension tablefunc;

create temp table paper_coauthors as (
		select type, paperid, count (distinct authid) as num_coauth
		from paperauths, papers, venue
		where paperauths.paperid = papers.id
		and papers.venue = venue.id
		group by paperid, type);

create temp table papers_count_per_type_per_num_coath as(
	select type, num_coauth, count(distinct paperid) as num_paper
	from paper_coauthors
	group by num_coauth, type);

insert into papers_count_per_type_per_num_coath
	select type, num_coauth, 0
	from
		(select distinct type from paper_coauthors) t,
		(select distinct num_coauth from paper_coauthors) n
	where not exists (
		select 1
		from papers_count_per_type_per_num_coath pcptptc
		where pcptptc.type = t.type
		and pcptptc.num_coauth = n.num_coauth
		limit 1
		);

select *
from crosstab('select num_coauth, type as type, sum(num_paper) as num_paper
			   from papers_count_per_type_per_num_coath 
			   group by num_coauth, type
			   order by num_coauth asc, type asc')
as ct (
  num_coauth bigint,
  journal_articles_publications numeric,
  conference_and_workshop_papers_publications numeric,
  books_and_thesis_publications numeric);