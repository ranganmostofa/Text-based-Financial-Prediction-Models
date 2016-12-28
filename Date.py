from datetime import datetime


NUM_DAYS = {"January": 31, "February": 28, "March": 31, "April": 30, "May": 31, "June": 30,
            "July": 31, "August": 31, "September": 30, "October": 31, "November": 30, "December": 31}

MONTHS = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
          7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}


class Date:
    """
    Date class that is used by the yahoo crawler to process the dates on the Yahoo!
    Finance website
    """
    def __init__(self, date):
        """
        Constructor of the Date class. Given the date represented as a string
        in the form "mm/dd/yy", creates an object of the Date class to store the
        date
        """
        self.date = date  # store the original representation
        # use string splicing to extract the day, month and year
        self.day = str(self.date).split("/")[1]
        self.month = str(self.date).split("/")[0]
        self.year = str(self.date).split("/")[2]

    def __str__(self):
        """
        Returns the original representation of the date as a string
        """
        return str(self.date)  # convert the date to a string and return it

    def get_date(self):
        """
        Returns the original representation of the date
        """
        return self.date  # return the date

    def get_day(self):
        """
        Returns the day in the date
        """
        return self.day  # return the day

    def get_month(self):
        """
        Returns the month in the date
        """
        return self.month  # return the month

    def get_year(self):
        """
        Returns the year in the date
        """
        year = self.year  # return the year

        current_year = str(datetime.now()).split(" ")[0].split("-")[0]  # obtain the current year

        # if the last 2 digits of the current year is greater than the 2 digits in the date
        if year > current_year[len(current_year) - 2:len(current_year)]:
            # the year must be from the 20th century
            year = "19" + year
        else:
            # otherwise the year is from the current / 21st century
            year = "20" + year

        return year  # return the year

    def set_date(self, new_date):
        """
        Changes the date to a user specified value
        """
        self.date = str(new_date)  # store the new date in the date field

    def set_day(self, day):
        """
        Changes the day in the date to a user specified value
        """
        self.day = str(day)  # store the new day in the day field

    def set_month(self, month):
        """
        Changes the month in the date to a user specified value
        """
        self.month = str(month)  # store the new month in the month field

    def set_year(self, year):
        """
        Changes the year in the date to a user specified value
        """
        self.year = str(year)  # store the new year in the year field

    def is_leap_year(self):
        """
        Returns true/false depending on whether the year in the date is a
        leap year or not
        """
        year = int(self.get_year())  # get the year in the date
        # if the year is divisible by 4
        if not year % 4:
            # if the year is divisible by both 100 and 400
            if not year % 100 and not year % 400:
                return True  # leap year
            # if the year is divisible by 100 but not 400
            if not year % 100 and year % 400:
                return False  # not a leap year
            # if the year is not divisible by 100
            if year % 100:
                return True  # leap year
        return False  # otherwise, all cases are not leap years

    def equals(self, date_obj):
        """
        Given a date object representing a single date, returns true if the date
        equals the date represented by the self date object and false otherwise
        """
        # return a boolean based on the equality of the two dates
        return self.get_date() == date_obj.get_date()

    def is_earlier(self, date_obj):
        """
        Given a date object representing a single date, returns true if the date
        comes earlier than the date represented by the self date object and false
        otherwise
        """
        if int(self.get_year()) < int(date_obj.get_year()):
            # if the year is lower in value, the self date object definitely comes earlier
            return True
        if int(self.get_month()) < int(date_obj.get_month()):
            # if the month is lower in value, the self date object definitely comes earlier
            return True
        if int(self.get_day()) < int(date_obj.get_day()):
            # if the day is lower in value, the self date object definitely comes earlier
            return True
        # if all tests have failed, the self date object must come later or is equal to the
        # given date object
        return False

    def is_later(self, date_obj):
        """
        Given a date object representing a single date, returns true if the date
        comes later than the date represented by the self date object and false otherwise
        """
        if int(self.get_year()) > int(date_obj.get_year()):
            # if the year is higher in value, the self date object definitely comes later
            return True
        if int(self.get_month()) > int(date_obj.get_month()):
            # if the month is higher in value, the self date object definitely comes later
            return True
        if int(self.get_day()) > int(date_obj.get_day()):
            # if the day is higher in value, the self date object definitely comes later
            return True
        # if all tests have failed, the self date object must come earlier or is equal to the
        # given date object
        return False

    def advance_date(self):
        """
        Advances the date stored in the date field to the next date
        """
        month_name = MONTHS[int(self.get_month())]  # obtain the name of the month

        if NUM_DAYS[month_name] <= int(self.get_day()):  # if the month is over
            if month_name == "December":  # if it is also the end of the year
                year = int(self.get_year()[len(self.get_year()) - 2:len(self.get_year())])
                year = (year + 1) % 100  # change the year to the next one
                year = str(year)
                if len(year) == 1:
                    year = "0" + year  # fixing the strings to suit the format
                day = "1"  # first day of the next month
                month = "1"  # first month of the next year
                # update the day, month and year fields of the class
                self.set_day(day), self.set_month(month), self.set_year(year)
            # if it is the 28th day of February of a leap year
            elif month_name == "February" and self.is_leap_year() and self.get_day() == "28":
                day = "29"  # February has an extra day
                self.set_day(day)  # update the day field of the class
            # if it is any other month
            else:
                day = "1"  # first day of the next month
                month = int(self.get_month())
                month += 1  # change the month to the next one
                # update the day and month fields of the class
                self.set_day(day), self.set_month(month)
        # if none of the above applies
        else:
            day = int(self.get_day())
            day += 1  # change only the next day
            # update only the day field of the class
            self.set_day(day)

        # create a string representing the date in the mm/dd/yy format
        new_date = self.get_month() + "/" + self.get_day() + "/" + \
                   self.get_year()[len(self.get_year()) - 2:len(self.get_year())]

        # update the date field of the class
        self.set_date(new_date)

    def url_format(self):
        """
        Returns a string representing the date in the url format, i.e. "yyyy-mm-dd"
        For example, 1/5/16 would be converted to 2016-01-05
        """
        day = self.get_day()  # get the stored day
        month = self.get_month()  # get the stored month
        year = self.get_year()  # get the stored year

        if len(month) == 1:
            month = "0" + month  # fixing month string to suit the format

        if len(day) == 1:
            day = "0" + day  # fixing day string to suit the format

        formatted_date = year + "-" + month + "-" + day  # create the string representation

        return formatted_date  # return the date as a string in the required url format

    def news_heading_format(self):
        """
        Returns a string representing the date in the news heading format, i.e.
        "(insert month name) day, year". For example 1/5/16 would be converted to
        January 5, 2016
        """
        day = self.get_day()  # get the stored day
        month = self.get_month()  # get the stored month
        year = self.get_year()  # get the stored year

        # create the string representation
        formatted_date = MONTHS[int(month)] + " " + day + ", " + year

        return formatted_date  # return the date as a string in the required url format


