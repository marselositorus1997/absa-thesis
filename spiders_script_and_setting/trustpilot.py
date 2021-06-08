import scrapy


class Trustpilot2Spider(scrapy.Spider):
    name = 'trustpilot2'
    allowed_domains = ['trustpilot.com']
    start_urls = ['https://www.trustpilot.com/review/www.decathlon.de']



    def parse(self, response):
        for div in response.xpath('//section/div[@class = "review-content"]'):
            yield {
                'company': response.xpath('//h1/span[1]/text()').extract_first().strip(), 
                'title': div.xpath('string(div/h2/a/text())').get().strip(),
                'main_comment': " ".join(div.xpath('string(div/p[descendant-or-self::text()])').get().split()),
                'invited_or_not': 1 if div.xpath('div[1]/div[1]/div[3]/script') else 0,
                'rating': div.xpath('div//@alt').re(r'(\d+)') 
            }
        
        #next page
        next_page = response.css('a.button.button--primary.next-page').attrib['href'] 
        if next_page:
            complete_next_url = response.urljoin(next_page)
            yield scrapy.Request(complete_next_url, callback=self.parse)


